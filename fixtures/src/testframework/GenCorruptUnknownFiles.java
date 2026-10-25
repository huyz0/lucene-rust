import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.CRC32;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.ByteArrayDataInput;
import org.apache.lucene.store.ByteBuffersDataOutput;
import java.util.Set;
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
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoDeletionPolicy;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.store.MockDirectoryWrapper;

/** Actual selection only: the corruption callback records its victims without random damage. */
public class GenCorruptUnknownFiles {
  static final StringBuilder OUT = new StringBuilder();
  static final HexFormat HEX = HexFormat.of();
  static void row(Object... values) {
    for (int i = 0; i < values.length; i++) { if (i != 0) OUT.append('\t'); OUT.append(values[i]); }
    OUT.append('\n');
  }
  static String names(Collection<String> names) {
    List<String> encoded = new ArrayList<>();
    for (String name : new TreeSet<>(names)) encoded.add(HEX.formatHex(name.getBytes(StandardCharsets.UTF_8)));
    return String.join(",", encoded);
  }
  static Map<String, byte[]> source() throws IOException {
    return source("fixtures/data/merge_policies/commits_index");
  }
  static Map<String, byte[]> source(String relativePath) throws IOException {
    Map<String, byte[]> files = new TreeMap<>();
    try (var paths = Files.list(Path.of(relativePath))) {
      for (Path path : paths.sorted().toList()) if (Files.isRegularFile(path)) files.put(path.getFileName().toString(), Files.readAllBytes(path));
    }
    return files;
  }
  static ByteBuffersDirectory directory(Map<String, byte[]> files) throws IOException {
    ByteBuffersDirectory dir = new ByteBuffersDirectory();
    for (var file : files.entrySet()) try (IndexOutput out = dir.createOutput(file.getKey(), IOContext.DEFAULT)) {
      out.writeBytes(file.getValue(), file.getValue().length);
    }
    return dir;
  }
  static byte[] bytes(Directory dir, String name) throws IOException {
    try (IndexInput in = dir.openInput(name, IOContext.DEFAULT)) {
      byte[] bytes = new byte[Math.toIntExact(in.length())]; in.readBytes(bytes, 0, bytes.length); return bytes;
    }
  }
  static final class Recording extends MockDirectoryWrapper {
    int calls;
    final TreeSet<String> victims = new TreeSet<>();
    Recording(Directory dir) {
      super(new Random(31), dir);
      setCheckIndexOnClose(false);
      setAssertNoUnrefencedFilesOnClose(false);
      setUseSlowOpenClosers(false);
      setThrottling(Throttling.NEVER);
    }
    @Override public synchronized void corruptFiles(Collection<String> files) {
      calls++; victims.addAll(files);
    }
  }
  static String run(String label, Map<String, byte[]> files) throws IOException {
    try (Recording dir = new Recording(directory(files))) {
      String result = "ok";
      try { dir.corruptUnknownFiles(); } catch (IOException | RuntimeException expected) { result = "failure"; }
      row("case", label, result, dir.calls, names(dir.victims));
      return result;
    }
  }
  static Map<String, byte[]> extras() {
    Map<String, byte[]> files = new TreeMap<>();
    for (String name : new String[] {"_a.doc", "_abc_suffix.ext", "_0.", "_0___.x", "_a..", "_a_.",
        "_A.doc", "_a-b.doc", "a.doc", "_afoo", "_0_suffix", "readme", "write.lock", "otherwrite.lock",
        "_a.write.lock", "_a.notwrite.lock", "_0_x\nz.foo", "_0.x\nz", "_0.x\r", "_0.x\u0085", "_0.x\u2028", "_0.x\u2029",
        "pending_segments", "pending_segments_1", "pending_segmentsAnything", "pending_segments_tail\n", "pending_segmentswrite.lock"}) {
      files.put(name, new byte[0]);
    }
    return files;
  }
  // Locate fields by using Lucene's own DataInput, then repair the real file's
  // checksum after a targeted negative mutation. No parser is copied to Rust.
  static ByteArrayDataInput body(byte[] bytes) throws IOException {
    ByteArrayDataInput in = new ByteArrayDataInput(bytes);
    in.skipBytes(4); in.readString(); in.skipBytes(4 + 16);
    in.skipBytes(Byte.toUnsignedInt(in.readByte()));
    return in;
  }
  static byte[] checked(byte[] bytes) {
    CRC32 crc = new CRC32(); crc.update(bytes, 0, bytes.length - 8);
    ByteBuffer.wrap(bytes).putLong(bytes.length - 8, crc.getValue());
    return bytes;
  }
  static byte[] integer(byte[] bytes, int offset, int value, ByteOrder order) {
    byte[] copy = bytes.clone(); ByteBuffer.wrap(copy).order(order).putInt(offset, value);
    return checked(copy);
  }
  record CommitOffsets(int minimum, int hard, int soft) {}
  static CommitOffsets commitOffsets(byte[] bytes) throws IOException {
    ByteArrayDataInput in = body(bytes);
    in.readVInt(); in.readVInt(); in.readVInt(); in.readVInt();
    in.skipBytes(8); in.readVLong();
    if (CodecUtil.readBEInt(in) != 3) throw new AssertionError("expected three segments");
    int minimum = in.getPosition();
    in.readVInt(); in.readVInt(); in.readVInt();
    if (!in.readString().equals("_0")) throw new AssertionError("expected first segment _0");
    in.skipBytes(16); in.readString(); in.skipBytes(8);
    int hard = in.getPosition(); in.skipBytes(4 + 8 + 8);
    return new CommitOffsets(minimum, hard, in.getPosition());
  }
  static void cross(String label, Map<String, byte[]> all, Map<String, byte[]> changes, String expected) throws IOException {
    Map<String, byte[]> files = new TreeMap<>(all);
    files.remove("segments_1"); files.remove("segments_2");
    try (ByteBuffersDirectory verified = directory(changes)) {
      for (String name : changes.keySet()) {
        if (!name.startsWith("segments") && !name.endsWith(".si")) continue;
        try (IndexInput input = verified.openInput(name, IOContext.DEFAULT)) { CodecUtil.checksumEntireFile(input); }
      }
    }
    for (var entry : new TreeMap<>(changes).entrySet()) {
      files.put(entry.getKey(), entry.getValue());
      row("replace", label, entry.getKey(), HEX.formatHex(entry.getValue()));
    }
    if (!run(label, files).equals(expected)) throw new AssertionError("unexpected cross-check outcome " + label);
  }
  static void crossChecks(Map<String, byte[]> all) throws IOException {
    byte[] commit = all.get("segments_3"); CommitOffsets offsets = commitOffsets(commit);
    cross("cross_hard_above_max", all, Map.of("segments_3", integer(commit, offsets.hard(), 3, ByteOrder.BIG_ENDIAN)), "failure");
    cross("cross_soft_above_max", all, Map.of("segments_3", integer(commit, offsets.soft(), 3, ByteOrder.BIG_ENDIAN)), "failure");
    byte[] sum = integer(commit, offsets.hard(), 2, ByteOrder.BIG_ENDIAN);
    sum = integer(sum, offsets.soft(), 1, ByteOrder.BIG_ENDIAN);
    cross("cross_sum_above_max", all, Map.of("segments_3", sum), "failure");
    cross("cross_sum_at_max", all, Map.of("segments_3", integer(commit, offsets.soft(), 1, ByteOrder.BIG_ENDIAN)), "ok");
    byte[] minimum = commit.clone(); minimum[offsets.minimum() + 1] = 6;
    cross("cross_minimum_above_segment", all, Map.of("segments_3", checked(minimum)), "failure");

    byte[] si = all.get("_0.si"); int version = body(si).getPosition();
    if (si[version + 12] != 1) throw new AssertionError("expected recorded minVersion");
    int minVersion = version + 13; int maxDoc = minVersion + 12;
    byte[] old = integer(si, version, 9, ByteOrder.LITTLE_ENDIAN);
    old = integer(old, minVersion, 9, ByteOrder.LITTLE_ENDIAN);
    byte[] loweredMinimum = commit.clone(); loweredMinimum[offsets.minimum()] = 9;
    loweredMinimum[offsets.minimum() + 1] = 0;
    cross("cross_segment_before_created", all, Map.of("segments_3", checked(loweredMinimum), "_0.si", old), "failure");
    cross("cross_minimum_version_too_old", all, Map.of("_0.si", integer(si, minVersion, 8, ByteOrder.LITTLE_ENDIAN)), "failure");
    byte[] absent = new byte[si.length - 12];
    System.arraycopy(si, 0, absent, 0, version + 12); absent[version + 12] = 0;
    System.arraycopy(si, minVersion + 12, absent, version + 13, si.length - minVersion - 12);
    cross("cross_minimum_version_absent", all, Map.of("_0.si", checked(absent)), "failure");

    Map<String, byte[]> oversized = new TreeMap<>();
    for (String name : new String[] {"_0.si", "_1.si", "_2.si"}) {
      byte[] original = all.get(name); int count = body(original).getPosition() + 25;
      oversized.put(name, integer(original, count, 1_000_000_000, ByteOrder.LITTLE_ENDIAN));
    }
    cross("cross_total_above_max", all, oversized, "failure");
    Map<String, byte[]> boundary = new TreeMap<>();
    boundary.put("_0.si", integer(si, maxDoc, IndexWriter.MAX_DOCS - 2, ByteOrder.LITTLE_ENDIAN));
    cross("cross_total_at_max", all, boundary, "ok");
  }
  static byte[] replaceRange(byte[] original, int start, int end, byte[] replacement) {
    byte[] copy = new byte[original.length - (end - start) + replacement.length];
    System.arraycopy(original, 0, copy, 0, start);
    System.arraycopy(replacement, 0, copy, start, replacement.length);
    System.arraycopy(original, end, copy, start + replacement.length, original.length - end);
    return checked(copy);
  }
  static byte[] siFiles(byte[] original, Set<String> additions, boolean omitSelf) throws IOException {
    ByteArrayDataInput input = body(original); input.skipBytes(12);
    if (input.readByte() != 1) throw new AssertionError("expected minVersion");
    input.skipBytes(12 + 4 + 2); input.readMapOfStrings();
    int start = input.getPosition(); TreeSet<String> files = new TreeSet<>(input.readSetOfStrings());
    int end = input.getPosition(); files.addAll(additions);
    if (omitSelf && !files.remove("_0.si")) throw new AssertionError("expected self in original SI files");
    ByteBuffersDataOutput output = new ByteBuffersDataOutput(); output.writeSetOfStrings(files);
    return replaceRange(original, start, end, output.toArrayCopy());
  }
  static void filenameChecks(Map<String, byte[]> all) throws IOException {
    byte[] original = all.get("_0.si");
    Map<String, byte[]> renamed = new TreeMap<>();
    renamed.put("_0.si", siFiles(original, Set.of("_a.norm", "_a_suffix.norm"), false));
    for (String name : new String[] {"_a.norm", "_0.norm", "_a_suffix.norm", "_0_suffix.norm"}) renamed.put(name, new byte[0]);
    cross("cross_si_normalized_names", all, renamed, "ok");
    cross("cross_si_omits_self", all, Map.of("_0.si", siFiles(original, Set.of(), true)), "ok");
    Map<String, String> invalid = new TreeMap<>();
    invalid.put("invalid_name", "notes"); invalid.put("tmp_name", "_a.TmP"); invalid.put("line_terminator", "_a.x\n");
    for (var bad : invalid.entrySet()) cross("cross_si_" + bad.getKey(), all, Map.of("_0.si", siFiles(original, Set.of(bad.getValue()), false)), "failure");

    byte[] commit = all.get("segments_3"); CommitOffsets offsets = commitOffsets(commit);
    ByteArrayDataInput input = new ByteArrayDataInput(commit); input.setPosition(offsets.soft() + 4);
    int marker = Byte.toUnsignedInt(input.readByte());
    if (marker == 1) input.skipBytes(16); else if (marker != 0) throw new AssertionError("SCI id marker");
    int start = input.getPosition();
    if (!input.readSetOfStrings().isEmpty() || CodecUtil.readBEInt(input) != 0) throw new AssertionError("expected no updates");
    int end = input.getPosition();
    Set<String> fieldFiles = new TreeSet<>(Set.of("_a.field", "_a_suffix.field", "_a.TMP", "plain"));
    Set<String> dvFiles = new TreeSet<>(Set.of("_b.dv", "_b_suffix.dv"));
    ByteBuffersDataOutput output = new ByteBuffersDataOutput(); output.writeSetOfStrings(fieldFiles);
    CodecUtil.writeBEInt(output, 1); CodecUtil.writeBEInt(output, 0); output.writeSetOfStrings(dvFiles);
    Map<String, byte[]> updates = new TreeMap<>();
    updates.put("segments_3", replaceRange(commit, start, end, output.toArrayCopy()));
    for (String name : fieldFiles) { updates.put(name, new byte[0]); updates.put("_0" + IndexFileNames.stripSegmentName(name), new byte[0]); }
    for (String name : dvFiles) { updates.put(name, new byte[0]); updates.put("_0" + IndexFileNames.stripSegmentName(name), new byte[0]); }
    cross("cross_commit_normalized_update_names", all, updates, "ok");

    byte[] empty = all.get("segments_4"); ByteArrayDataInput header = body(empty);
    int writerMajor = header.getPosition(); header.readVInt(); header.readVInt(); header.readVInt(); int createdMajor = header.getPosition();
    byte[] tooOld = empty.clone(); tooOld[createdMajor] = 5;
    cross("cross_empty_created_too_old", all, Map.of("segments_4", checked(tooOld)), "failure");
    byte[] future = empty.clone(); future[writerMajor] = 11; future[createdMajor] = 11;
    cross("cross_empty_created_future", all, Map.of("segments_4", checked(future)), "failure");
  }
  static void simpleRun(String label, Map<String, byte[]> files, String expected) throws IOException {
    if (!run(label, files).equals(expected)) throw new AssertionError("unexpected SimpleText outcome " + label);
  }
  static void simpleTextChecks() throws IOException {
    Map<String, byte[]> retained = new TreeMap<>();
    for (String stage : new String[] {"flush", "merged"}) {
      Map<String, byte[]> clean = source("fixtures/data/codecs_simpletext/codec/" + stage + "/index");
      retained.putAll(clean);
      String prefix = "simpletext_" + stage + "_";
      String si = clean.keySet().stream().filter(name -> name.endsWith(".si")).findFirst().orElseThrow();
      String commit = clean.keySet().stream().filter(name -> name.startsWith("segments_")).findFirst().orElseThrow();
      String live = clean.keySet().stream().filter(name -> name.endsWith(".liv")).findFirst().orElseThrow();
      simpleRun(prefix + "clean", clean, "ok");
      Map<String, byte[]> extra = new TreeMap<>(clean); extra.putAll(extras());
      simpleRun(prefix + "extras", extra, "ok");
      Map<String, byte[]> missing = new TreeMap<>(extra); missing.remove(si);
      simpleRun(prefix + "missing_si", missing, "failure");
      Map<String, byte[]> corruptSi = new TreeMap<>(extra); corruptSi.put(si, new byte[] {1, 2, 3});
      simpleRun(prefix + "corrupt_si", corruptSi, "failure");
      Map<String, byte[]> corruptCommit = new TreeMap<>(extra); corruptCommit.put(commit, new byte[] {1, 2, 3});
      simpleRun(prefix + "corrupt_commit", corruptCommit, "failure");
      // Selection reads commit and SI metadata, not the referenced live-doc bytes.
      Map<String, byte[]> corruptLive = new TreeMap<>(extra); corruptLive.put(live, new byte[] {1, 2, 3});
      simpleRun(prefix + "corrupt_live", corruptLive, "ok");
      Map<String, byte[]> missingLive = new TreeMap<>(extra); missingLive.remove(live);
      simpleRun(prefix + "missing_live", missingLive, "ok");
    }
    simpleRun("simpletext_retained_clean", retained, "ok");
    Map<String, byte[]> extra = new TreeMap<>(retained); extra.putAll(extras());
    simpleRun("simpletext_retained_extras", extra, "ok");
    Map<String, byte[]> latest = new TreeMap<>(extra); latest.remove("segments_2");
    simpleRun("simpletext_retained_latest_only", latest, "ok");
    Map<String, byte[]> oldest = new TreeMap<>(extra); oldest.remove("segments_5");
    simpleRun("simpletext_retained_oldest_only", oldest, "ok");
  }
  public static void main(String[] args) throws Exception {
    // Must precede every Lucene initialization: the sole new commit ID is reproducible.
    System.setProperty("tests.seed", "0");
    Map<String, byte[]> source = source();
    Map<String, byte[]> retained = new TreeMap<>(source);
    // Existing committed Java bytes avoid all fresh .si diagnostics/timestamps.
    // deleteAll creates only an empty commit, making every codec file old-only.
    try (ByteBuffersDirectory dir = directory(source)) {
      IndexWriterConfig config = new IndexWriterConfig().setOpenMode(IndexWriterConfig.OpenMode.APPEND)
          .setIndexDeletionPolicy(NoDeletionPolicy.INSTANCE).setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter writer = new IndexWriter(dir, config)) { writer.deleteAll(); writer.commit(); }
      TreeSet<String> newNames = new TreeSet<>(List.of(dir.listAll())); newNames.removeAll(source.keySet());
      if (!newNames.equals(new TreeSet<>(List.of("segments_4")))) throw new AssertionError("unexpected new files " + newNames);
      byte[] snapshot = bytes(dir, "segments_4"); retained.put("segments_4", snapshot);
      row("snapshot", "segments_4", HEX.formatHex(snapshot));
      TreeSet<String> protectedFiles = new TreeSet<>();
      for (String name : source.keySet()) if (name.startsWith("segments_")) protectedFiles.addAll(SegmentInfos.readCommit(dir, name).files(true));
      protectedFiles.removeAll(SegmentInfos.readCommit(dir, "segments_4").files(true));
      protectedFiles.removeIf(name -> !IndexFileNames.CODEC_FILE_PATTERN.matcher(name).matches());
      if (protectedFiles.isEmpty()) throw new AssertionError("must exercise old-only codec files");
      row("proof", "old_only", names(protectedFiles));
    }
    run("empty", Map.of());
    run("name_edges", extras());
    run("retained_clean", retained);
    Map<String, byte[]> all = new TreeMap<>(retained); all.putAll(extras()); run("retained_extras", all);
    Map<String, byte[]> latest = new TreeMap<>(all); latest.keySet().removeIf(name -> name.startsWith("segments_") && !name.equals("segments_4")); run("latest_only", latest);
    Map<String, byte[]> oldest = new TreeMap<>(source); oldest.remove("segments_2"); oldest.remove("segments_3"); oldest.putAll(extras()); run("oldest_only", oldest);
    for (String alias : new String[] {"segmentsX4", "segments§4", "segments_+4", "segments_٤", "segments_４", "segments_００４"}) {
      Map<String, byte[]> aliased = new TreeMap<>(all); byte[] snapshot = aliased.remove("segments_4");
      aliased.put(alias, snapshot); run("alias_" + HEX.formatHex(alias.getBytes(StandardCharsets.UTF_8)), aliased);
    }
    for (String bad : new String[] {"segments", "segments4", "segments🦀4", "segments_bad", "segments.gen", "segments_?"}) {
      Map<String, byte[]> broken = new TreeMap<>(all); broken.put(bad, new byte[] {1, 2, 3}); run("bad_" + HEX.formatHex(bad.getBytes(StandardCharsets.UTF_8)), broken);
    }
    Map<String, byte[]> missing = new TreeMap<>(all); missing.remove("_0.si"); run("missing_si", missing);
    Map<String, byte[]> corrupt = new TreeMap<>(all); corrupt.put("_0.si", new byte[] {1, 2, 3}); run("corrupt_si", corrupt);
    crossChecks(all);
    filenameChecks(all);
    simpleTextChecks();
    Path out = Path.of(args[0]); Files.createDirectories(out); Files.writeString(out.resolve("corrupt_unknown_files.tsv"), OUT.toString(), StandardCharsets.UTF_8);
  }
}
