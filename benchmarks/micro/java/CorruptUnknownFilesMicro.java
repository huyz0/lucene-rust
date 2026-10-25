import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Random;
import java.util.TreeSet;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.store.MockDirectoryWrapper;

/** Actual corruptUnknownFiles selection; the damage callback records victims without mutation. */
public final class CorruptUnknownFilesMicro {
  private static final long BASIS = 0xcbf29ce484222325L;
  private static volatile long sink;
  private record Result(long digest, int count) {}
  private static Result digest(Collection<String> files) {
    long hash = BASIS;
    for (String name : new TreeSet<>(files)) {
      for (byte b : name.getBytes(StandardCharsets.UTF_8)) {
        hash = (hash ^ Byte.toUnsignedInt(b)) * 0x100000001b3L;
      }
      hash = (hash ^ 0) * 0x100000001b3L;
    }
    return new Result(hash, files.size());
  }
  private static final class RecordingDirectory extends MockDirectoryWrapper {
    Result result;
    RecordingDirectory(Directory delegate) {
      super(new Random(42), delegate);
      setCheckIndexOnClose(false);
      setAssertNoUnrefencedFilesOnClose(false);
      setUseSlowOpenClosers(false);
      setThrottling(Throttling.NEVER);
    }
    @Override
    public synchronized void corruptFiles(Collection<String> files) {
      result = digest(files);
    }
  }
  private static String orphan(int i) {
    return i % 2 == 0 ? String.format(java.util.Locale.ROOT, "_orphan%06d.si", i)
        : String.format(java.util.Locale.ROOT, "pending_segments_orphan%06d", i);
  }
  private static long[] run(RecordingDirectory dir, Result expected, long millis) throws Exception {
    long deadline = System.nanoTime() + millis * 1_000_000L;
    long nanos = 0;
    long count = 0;
    do {
      long start = System.nanoTime();
      dir.corruptUnknownFiles();
      nanos += System.nanoTime() - start;
      if (!expected.equals(dir.result)) {
        throw new AssertionError("victim selection mismatch: " + dir.result + " != " + expected);
      }
      sink = dir.result.digest;
      count++;
    } while (System.nanoTime() < deadline);
    return new long[] {nanos, count};
  }
  public static void main(String[] args) throws Exception {
    Path snapshot = Path.of(args[0]);
    long warmup = Long.parseLong(args[1]);
    long measure = Long.parseLong(args[2]);
    if (warmup < 0 || measure <= 0) throw new IllegalArgumentException("invalid duration");
    for (int orphans : new int[] {8, 2048}) {
      ByteBuffersDirectory memory = new ByteBuffersDirectory();
      try (Directory source = FSDirectory.open(snapshot)) {
        for (String name : source.listAll()) memory.copyFrom(source, name, name, IOContext.DEFAULT);
      }
      TreeSet<String> victims = new TreeSet<>();
      for (int i = 0; i < orphans; i++) victims.add(orphan(i));
      for (String name : victims) {
        try (var output = memory.createOutput(name, IOContext.DEFAULT)) { output.writeByte((byte) 1); }
      }
      for (String name : new String[] {"README", "_guardwrite.lock"}) {
        try (var output = memory.createOutput(name, IOContext.DEFAULT)) { output.writeByte((byte) 1); }
      }
      Result expected = digest(victims);
      try (RecordingDirectory dir = new RecordingDirectory(memory)) {
        run(dir, expected, warmup);
        long[] measured = run(dir, expected, measure);
        String name = orphans == 8 ? "corrupt_unknown_small" : "corrupt_unknown_large";
        System.out.printf(java.util.Locale.ROOT, "%s\t%.3f\t%d%n", name,
            (double) measured[0] / measured[1], measured[1]);
        System.out.printf("#check\t%s\t%s\t%d%n", name,
            Long.toUnsignedString(expected.digest), expected.count);
      }
    }
  }
}
