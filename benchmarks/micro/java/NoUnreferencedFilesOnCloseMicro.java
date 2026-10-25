import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.store.MockDirectoryWrapper;

/** Actual close, including its two writer passes and document-count comparison. */
public final class NoUnreferencedFilesOnCloseMicro {
  private static volatile long sink;
  private record Expected(TreeSet<String> files, int docs, long digest, String error) {}
  private record Result(long nanos, long count) {}
  private static long digest(TreeSet<String> files, int docs) {
    long hash = 0xcbf29ce484222325L;
    for (String name : files) {
      for (byte value : name.getBytes(StandardCharsets.UTF_8)) {
        hash = (hash ^ Byte.toUnsignedInt(value)) * 0x100000001b3L;
      }
      hash *= 0x100000001b3L;
    }
    return (hash ^ docs) * 0x100000001b3L;
  }
  // MockDirectoryWrapper still calls its delegate's close. Keep only the memory
  // store alive so verification and backing-store cleanup are outside timing,
  // matching Rust's Directory (whose trait has no close).
  private static final class RecordingClose extends FilterDirectory {
    int closes;
    RecordingClose(ByteBuffersDirectory in) { super(in); }
    @Override public void close() { closes++; }
  }
  private static Map<String, byte[]> source(Path path) throws IOException {
    Map<String, byte[]> files = new TreeMap<>();
    try (var paths = Files.list(path)) {
      for (Path file : paths.sorted().toList()) {
        files.put(file.getFileName().toString(), Files.readAllBytes(file));
      }
    }
    return files;
  }
  private static Result run(Map<String, byte[]> files, Expected expected,
                            boolean retained, boolean checkPruning, long millis) throws Exception {
    long deadline = System.nanoTime() + millis * 1_000_000L;
    long nanos = 0;
    long count = 0;
    do {
      try (ByteBuffersDirectory memory = new ByteBuffersDirectory()) {
        for (var file : files.entrySet()) {
          try (var output = memory.createOutput(file.getKey(), IOContext.DEFAULT)) {
            output.writeBytes(file.getValue(), file.getValue().length);
          }
        }
        RecordingClose delegate = new RecordingClose(memory);
        MockDirectoryWrapper dir = new MockDirectoryWrapper(new Random(42), delegate);
        dir.setCheckIndexOnClose(false);
        dir.setAssertNoUnrefencedFilesOnClose(true);
        dir.setUseSlowOpenClosers(false);
        dir.setThrottling(MockDirectoryWrapper.Throttling.NEVER);
        long start = System.nanoTime();
        Exception failure = null;
        try { dir.close(); } catch (Exception error) { failure = error; }
        nanos += System.nanoTime() - start;
        if (expected.error.isEmpty()) {
          if (failure != null) throw new AssertionError("unexpected close failure", failure);
        } else if (!(failure instanceof IllegalArgumentException)
            || !expected.error.equals(failure.getMessage())) {
          throw new AssertionError("expected parent-field close failure: " + expected.error, failure);
        }
        if (delegate.closes != 1) throw new AssertionError("delegate close count");
        TreeSet<String> after = new TreeSet<>(Arrays.asList(memory.listAll()));
        if (!after.equals(expected.files)) throw new AssertionError("final file set: " + after);
        if (checkPruning && retained == files.keySet().equals(after)) {
          throw new AssertionError("retained commits must be pruned; latest must be unchanged");
        }
        int docs;
        try (DirectoryReader reader = DirectoryReader.open(memory)) { docs = reader.numDocs(); }
        if (docs != expected.docs) throw new AssertionError("document count: " + docs);
        long actual = digest(after, docs);
        if (actual != expected.digest) throw new AssertionError("digest mismatch");
        sink = actual;
      }
      count++;
    } while (System.nanoTime() < deadline);
    return new Result(nanos, count);
  }
  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]);
    long warmup = Long.parseLong(args[1]);
    long measure = Long.parseLong(args[2]);
    if (warmup < 0 || measure <= 0) throw new IllegalArgumentException("invalid duration");
    for (String stage : new String[] {"latest", "retained", "simpletext_flush", "simpletext_merged"}) {
      TreeSet<String> finalFiles = new TreeSet<>(Files.readAllLines(root.resolve(stage + "-expected-files.txt")));
      int docs = Integer.parseInt(Files.readString(root.resolve(stage + "-expected-docs.txt")).trim());
      String error = Files.readString(root.resolve(stage + "-expected-error.txt"));
      Expected expected = new Expected(finalFiles, docs, digest(finalFiles, docs), error);
      Map<String, byte[]> files = source(root.resolve(stage));
      boolean retained = stage.equals("retained");
      boolean checkPruning = !stage.startsWith("simpletext_");
      if (retained && !(files.containsKey("segments_1") && files.containsKey("segments_2"))) {
        throw new AssertionError("retained snapshot missing old commits");
      }
      run(files, expected, retained, checkPruning, warmup);
      Result result = run(files, expected, retained, checkPruning, measure);
      String name = "no_unreferenced_close_" + stage;
      System.out.printf(java.util.Locale.ROOT, "%s\t%.3f\t%d%n", name,
          (double) result.nanos / result.count, result.count);
      System.out.printf("#check\t%s\t%s\t%d\t%d\t%s%n", name,
          Long.toUnsignedString(expected.digest), expected.files.size(), expected.docs,
          expected.error.isEmpty() ? "ok" : "failure");
    }
  }
}
