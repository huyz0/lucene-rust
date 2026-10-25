import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoDeletionPolicy;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.Lock;
import org.apache.lucene.tests.store.MockDirectoryWrapper;

/** Actual close, including writer rollback and the subsequent doc-count assertion. */
public class GenNoUnreferencedFilesOnClose {
  static final HexFormat HEX = HexFormat.of();
  static final StringBuilder OUT = new StringBuilder();
  static void row(Object... fields) {
    for (int i = 0; i < fields.length; i++) { if (i != 0) OUT.append('\t'); OUT.append(fields[i]); }
    OUT.append('\n');
  }
  static String hex(String text) { return HEX.formatHex(text.getBytes(StandardCharsets.UTF_8)); }
  static String names(Collection<String> names) {
    List<String> encoded = new ArrayList<>();
    for (String name : new TreeSet<>(names)) encoded.add(hex(name));
    return String.join(",", encoded);
  }
  static Map<String, byte[]> source(String kind) throws IOException {
    Map<String, byte[]> files = new TreeMap<>();
    if (kind.equals("none")) return files;
    String path = kind.equals("parent_binary") ? "fixtures/data/misc/bp_blocks"
        : kind.equals("soft_delete_binary") ? "fixtures/data/field_infos_index"
        : kind.startsWith("simpletext_")
        ? "fixtures/data/codecs_simpletext/codec/" + kind.substring(11) + "/index"
        : "fixtures/data/merge_policies/commits_index";
    try (var paths = Files.list(Path.of(path))) {
      for (Path file : paths.sorted().toList()) {
        String name = file.getFileName().toString();
        if ((kind.equals("parent_binary") || kind.equals("soft_delete_binary"))
            && (name.equals("manifest.properties") || name.endsWith(".raw"))) continue;
        if (Files.isRegularFile(file)) files.put(name, Files.readAllBytes(file));
      }
    }
    if (kind.equals("latest")) { files.remove("segments_1"); files.remove("segments_2"); }
    if (kind.equals("empty_latest")) {
      String snapshot = Files.readAllLines(Path.of("fixtures/testframework/data/corrupt_unknown_files.tsv")).stream()
          .filter(line -> line.startsWith("snapshot\tsegments_4\t")).findFirst().orElseThrow();
      files.put("segments_4", HEX.parseHex(snapshot.split("\t")[2]));
    }
    return files;
  }
  static void write(Directory dir, String name, byte[] bytes) throws IOException {
    try (IndexOutput out = dir.createOutput(name, IOContext.DEFAULT)) { out.writeBytes(bytes, bytes.length); }
  }
  static int numDocs(Directory directory) {
    try (DirectoryReader reader = DirectoryReader.open(directory)) { return reader.numDocs(); }
    catch (IOException | RuntimeException expected) { return -1; }
  }
  static final class TraceDirectory extends FilterDirectory {
    boolean armed;
    int phase, faults, closes;
    final String operation;
    final Map<Integer, TreeSet<String>> removed = new TreeMap<>();
    final Map<Integer, TreeSet<String>> added = new TreeMap<>();
    TraceDirectory(Directory inner, String operation) { super(inner); this.operation = operation; }
    void fault(String location) throws IOException {
      if (armed && faults == 0 && operation.equals("fault_" + location)) {
        faults++; throw new IOException("fixture fault: " + location);
      }
    }
    @Override public String[] listAll() throws IOException { fault("listing"); return in.listAll(); }
    @Override public IndexInput openInput(String name, IOContext context) throws IOException {
      if (phase > 0 && name.endsWith(".si")) fault("read");
      return in.openInput(name, context);
    }
    @Override public void deleteFile(String name) throws IOException {
      if (name.equals("_z.orphan")) fault("delete");
      in.deleteFile(name);
      if (armed) removed.computeIfAbsent(phase, ignored -> new TreeSet<>()).add(name);
    }
    @Override public Lock obtainLock(String name) throws IOException {
      int acquiredPhase = armed ? ++phase : 0;
      fault("lock");
      Lock delegate = in.obtainLock(name);
      return new Lock() {
        boolean closed;
        @Override public void ensureValid() throws IOException { delegate.ensureValid(); }
        @Override public void close() throws IOException {
          if (closed) return;
          closed = true; delegate.close();
          if (armed && acquiredPhase == 1 && operation.equals("added_on_rollback")) {
            write(in, "_added.after", new byte[0]);
            added.computeIfAbsent(acquiredPhase, ignored -> new TreeSet<>()).add("_added.after");
          }
        }
      };
    }
    // Keep the delegate readable after actual MockDirectoryWrapper.close.
    @Override public void close() { closes++; }
  }
  static final class Recording extends MockDirectoryWrapper {
    int corruptionCalls;
    Recording(Directory inner) {
      super(new Random(31), inner);
      setUseSlowOpenClosers(false); setThrottling(Throttling.NEVER);
    }
    @Override public synchronized void corruptFiles(Collection<String> names) { corruptionCalls++; }
  }
  static Throwable close(Recording dir) {
    try { dir.close(); return null; } catch (Throwable expected) { return expected; }
  }
  static void outcome(String label, String attempt, Throwable error) {
    row("outcome", label, attempt, error == null ? "ok" : "failure",
        error == null ? "" : error.getClass().getSimpleName(), error == null ? "" : hex(String.valueOf(error.getMessage()).replaceAll("@[0-9a-fA-F]+", "@ID")));
  }
  static void run(String label, String kind, String assertion, boolean check, String operation) throws Exception {
    Map<String, byte[]> files = source(kind);
    if (operation.equals("orphan_codec") || operation.equals("fault_delete")) files.put("_z.orphan", new byte[] {1});
    if (operation.equals("orphan_pending")) files.put("pending_segments_orphan", new byte[] {1});
    if (operation.equals("decoys")) { files.put("README", new byte[0]); files.put("_guardwrite.lock", new byte[0]); }
    if (operation.equals("name_edges")) {
      for (String name : new String[] {"_z.x\n", "_z_x\r.foo", "_z.x\u0085", "_z.x\u2028", "_z.x\u2029", "_z.x\u000b", "_z.x\u000c", "pending_segments_tail\n"}) files.put(name, new byte[] {1});
    }
    if (operation.equals("exact_segments")) { files.put("segments", new byte[] {1, 2, 3}); files.put("_z.orphan", new byte[] {1}); }
    if (operation.equals("malformed_segments")) files.put("segments_?", new byte[] {1, 2, 3});
    if (operation.equals("corrupt_latest")) files.put("segments_3", new byte[] {1, 2, 3});
    if (operation.equals("corrupt_old")) files.put("segments_1", new byte[] {1, 2, 3});
    if (operation.equals("missing_si")) files.remove("_0.si");
    if (operation.equals("corrupt_si")) files.put("_0.si", new byte[] {1, 2, 3});
    if (operation.equals("missing_fnm")) files.remove("_0.fnm");
    if (operation.equals("corrupt_fnm")) files.put("_0.fnm", new byte[] {1, 2, 3});
    if (operation.equals("missing_fnm") || operation.equals("corrupt_fnm")) files.put("_z.orphan", new byte[] {1});
    if (operation.equals("missing_live")) files.remove("_0_1.liv");
    if (operation.equals("corrupt_live")) files.put("_0_1.liv", new byte[] {1, 2, 3});
    ByteBuffersDirectory inner = new ByteBuffersDirectory();
    for (var file : files.entrySet()) write(inner, file.getKey(), file.getValue());
    TraceDirectory trace = new TraceDirectory(inner, operation);
    Recording dir = new Recording(trace); dir.setCheckIndexOnClose(check);
    if (!assertion.equals("default")) dir.setAssertNoUnrefencedFilesOnClose(assertion.equals("on"));
    IndexInput leaked = operation.equals("leak_input") ? dir.openInput("_0.si", IOContext.DEFAULT) : null;
    Lock held = operation.equals("wrapper_lock") ? dir.obtainLock("write.lock")
        : operation.equals("delegate_lock") ? trace.obtainLock("write.lock") : null;
    int before = numDocs(inner);
    row("case", label, kind, assertion, check, operation);
    row("state", label, "before", names(files.keySet()), before);
    if (operation.equals("rates")) { dir.setRandomIOExceptionRate(1.0); dir.setRandomIOExceptionRateOnOpen(1.0); }
    trace.armed = true;
    Throwable first = close(dir); outcome(label, "first", first);
    row("state", label, "after", names(List.of(inner.listAll())), numDocs(inner));
    row("trace", label, trace.phase, trace.faults, dir.corruptionCalls, trace.closes, dir.isOpen(),
        dir.getRandomIOExceptionRate(), dir.getRandomIOExceptionRateOnOpen());
    for (int phase = 0; phase <= trace.phase; phase++) {
      row("removed", label, phase, names(trace.removed.getOrDefault(phase, new TreeSet<>())));
      row("added", label, phase, names(trace.added.getOrDefault(phase, new TreeSet<>())));
    }
    if (leaked != null) leaked.close(); if (held != null) held.close();
    Throwable second = close(dir); outcome(label, "second", second);
    row("repeat", label, trace.phase, trace.faults, dir.corruptionCalls, trace.closes,
        names(List.of(inner.listAll())), numDocs(inner));
    inner.close();
  }
  static void rollbackProbe(String operation) throws Exception {
    ByteBuffersDirectory inner = new ByteBuffersDirectory();
    for (var file : source("latest").entrySet()) write(inner, file.getKey(), file.getValue());
    TraceDirectory trace = new TraceDirectory(inner, "fault_listing");
    String before = names(List.of(inner.listAll()));
    Throwable error = null;
    // Arm after construction so the fault belongs to rollback's refresh,
    // independent of how many times the constructor lists the directory.
    try (IndexWriter writer = new IndexWriter(trace,
        new IndexWriterConfig(null).setIndexDeletionPolicy(NoDeletionPolicy.INSTANCE).setCommitOnClose(false))) {
      trace.armed = true;
      if (operation.equals("rollback")) writer.rollback();
      else if (operation.equals("close_without_commit")) writer.close();
      else throw new IllegalArgumentException(operation);
    } catch (Throwable caught) { error = caught; }
    boolean released = false;
    try (Lock ignored = inner.obtainLock("write.lock")) { released = true; }
    row("rollback_probe", operation, error == null ? "ok" : "failure",
        error == null ? "" : error.getClass().getSimpleName(),
        error == null ? "" : hex(String.valueOf(error.getMessage())),
        before, names(List.of(inner.listAll())), trace.faults, released);
    inner.close();
  }
  public static void main(String[] args) throws Exception {
    if (!GenNoUnreferencedFilesOnClose.class.desiredAssertionStatus()) throw new AssertionError("generator needs -ea");
    System.setProperty("tests.seed", "0");
    run("default_off", "retained", "default", false, "orphan_codec");
    run("explicit_off", "retained", "off", false, "orphan_codec");
    run("no_index_empty", "none", "on", false, "clean");
    run("no_index_orphan", "none", "on", false, "orphan_codec");
    run("no_index_exact_segments", "none", "on", false, "exact_segments");
    run("malformed_segments", "none", "on", false, "malformed_segments");
    run("clean_latest", "latest", "on", false, "clean");
    run("clean_retained", "retained", "on", false, "clean");
    run("clean_empty_latest", "empty_latest", "on", false, "clean");
    for (String operation : new String[] {"orphan_codec", "orphan_pending", "decoys", "name_edges", "rates", "corrupt_latest", "missing_si", "corrupt_si", "missing_fnm", "corrupt_fnm", "missing_live", "corrupt_live", "leak_input", "wrapper_lock", "delegate_lock", "fault_listing", "fault_lock", "fault_read", "fault_delete", "added_on_rollback"}) {
      run(operation, "latest", "on", false, operation);
    }
    run("corrupt_old", "retained", "on", false, "corrupt_old");
    run("both_checks_latest", "latest", "on", true, "clean");
    run("both_checks_retained", "retained", "on", true, "clean");
    run("simpletext_flush", "simpletext_flush", "on", false, "clean");
    run("simpletext_merged", "simpletext_merged", "on", false, "clean");
    run("parent_binary", "parent_binary", "on", false, "clean");
    run("soft_delete_binary", "soft_delete_binary", "on", false, "clean");
    rollbackProbe("rollback");
    rollbackProbe("close_without_commit");
    Path out = Path.of(args[0]); Files.createDirectories(out);
    Files.writeString(out.resolve("no_unreferenced_files_on_close.tsv"), OUT.toString(), StandardCharsets.UTF_8);
  }
}
