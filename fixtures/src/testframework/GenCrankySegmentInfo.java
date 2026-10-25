import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.SegmentInfoFormat;
import org.apache.lucene.codecs.lucene99.Lucene99SegmentInfoFormat;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Version;

/** Actual CrankySegmentInfoFormat delegation, shared RNG and .si bytes from Lucene 10.5.0. */
public class GenCrankySegmentInfo {
  static final byte[] ID = "0123456789abcdef".getBytes(StandardCharsets.US_ASCII);

  static SegmentInfo info(Directory dir, String name) {
    var info = new SegmentInfo(dir, Version.fromBits(10, 5, 0), null, name, 7,
        false, false, null, Map.of(), ID.clone(), Map.of(), null);
    info.setFiles(Set.of());
    return info;
  }

  static SegmentInfoFormat wrap(SegmentInfoFormat delegate, Random random) throws Exception {
    var type = Class.forName("org.apache.lucene.tests.codecs.cranky.CrankySegmentInfoFormat");
    var constructor = type.getDeclaredConstructor(SegmentInfoFormat.class, Random.class);
    constructor.setAccessible(true);
    return (SegmentInfoFormat) constructor.newInstance(delegate, random);
  }

  static class State {
    int reads;
    int writes;
    boolean failRead;
    boolean failWrite;
  }

  static class Recording extends SegmentInfoFormat {
    final State state;
    final SegmentInfoFormat delegate = new Lucene99SegmentInfoFormat();
    Recording(State state) { this.state = state; }
    @Override public SegmentInfo read(Directory dir, String name, byte[] id, IOContext context) throws IOException {
      state.reads++;
      if (state.failRead) throw new IOException("delegate read failure");
      return delegate.read(dir, name, id, context);
    }
    @Override public void write(Directory dir, SegmentInfo info, IOContext context) throws IOException {
      state.writes++;
      if (state.failWrite) throw new IOException("delegate write failure");
      delegate.write(dir, info, context);
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    try (var dir = new ByteBuffersDirectory()) {
      new Lucene99SegmentInfoFormat().write(dir, info(dir, "_base"), IOContext.DEFAULT);
      try (var input = dir.openInput("_base.si", IOContext.DEFAULT)) {
        byte[] bytes = new byte[(int) input.length()];
        input.readBytes(bytes, 0, bytes.length);
        Files.write(out.resolve("cranky_segment_info.si"), bytes);
      }
    }
    var trace = new StringBuilder();
    for (long seed : new long[] {0, 42, 777, Long.MIN_VALUE}) {
      for (boolean failures : new boolean[] {false, true}) {
        Random random = new Random(seed);
        State state = new State();
        SegmentInfoFormat[] formats = {wrap(new Recording(state), random), wrap(new Recording(state), random)};
        try (var dir = new ByteBuffersDirectory()) {
          new Lucene99SegmentInfoFormat().write(dir, info(dir, "_base"), IOContext.DEFAULT);
          for (int op = 0; op < 256; op++) {
            boolean read = op % 4 == 0;
            state.failRead = failures && op % 32 == 0;
            state.failWrite = failures && op % 19 == 0;
            String name = "_s" + Integer.toString(op, 36);
            SegmentInfo info = info(dir, name);
            String result = "ok";
            String docs = "-";
            try {
              if (read) docs = Integer.toString(formats[op % 2].read(dir, "_base", ID, IOContext.DEFAULT).maxDoc());
              else formats[op % 2].write(dir, info, IOContext.DEFAULT);
            } catch (IOException e) {
              result = e.getMessage();
            }
            trace.append("op\t").append(seed).append('\t').append(failures).append('\t').append(op)
                .append('\t').append(read ? "read" : "write").append('\t').append(result)
                .append('\t').append(state.reads).append('\t').append(state.writes)
                .append('\t').append(info.files().contains(name + ".si")).append('\t').append(docs).append('\n');
          }
          trace.append("rng\t").append(seed).append('\t').append(failures).append('\t').append(random.nextInt()).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("cranky_segment_info.tsv"), trace, StandardCharsets.UTF_8);
  }
}
