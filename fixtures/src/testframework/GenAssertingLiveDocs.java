import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import org.apache.lucene.codecs.LiveDocsFormat;
import org.apache.lucene.codecs.lucene90.Lucene90LiveDocsFormat;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.codecs.asserting.AssertingLiveDocsFormat;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.LiveDocs;
import org.apache.lucene.util.Version;

/** Actual asserting format/wrappers with assertions enabled, including sparse Java reads. */
public class GenAssertingLiveDocs {
  static final byte[] ID = "0123456789abcdef".getBytes(StandardCharsets.US_ASCII);
  static final StringBuilder OUT = new StringBuilder();
  static SegmentCommitInfo info(Directory dir, int length, long generation, int deleted) {
    SegmentInfo si = new SegmentInfo(dir, Version.LATEST, Version.LATEST, "_ald", length,
        false, false, null, Map.of(), ID, Map.of(), null);
    return new SegmentCommitInfo(si, deleted, 0, generation, -1, -1, null);
  }
  static String bits(Bits bits) {
    StringBuilder out = new StringBuilder();
    for (int i = 0; i < bits.length(); i++) out.append(bits.get(i) ? '1' : '0');
    return out.isEmpty() ? "-" : out.toString();
  }
  static void emit(Object... fields) {
    for (int i = 0; i < fields.length; i++) {
      if (i != 0) OUT.append('\t');
      OUT.append(fields[i]);
    }
    OUT.append('\n');
  }
  static Bits wrapBits(Bits raw) throws Exception {
    var type = Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingLiveDocsFormat$AssertingBits");
    var ctor = type.getDeclaredConstructor(Bits.class);
    ctor.setAccessible(true);
    return (Bits) ctor.newInstance(raw);
  }
  static LiveDocs wrapLiveDocs(LiveDocs raw) throws Exception {
    var type = Class.forName("org.apache.lucene.tests.codecs.asserting.AssertingLiveDocsFormat$AssertingLiveDocs");
    var ctor = type.getDeclaredConstructor(LiveDocs.class);
    ctor.setAccessible(true);
    return (LiveDocs) ctor.newInstance(raw);
  }
  static class Probe extends LiveDocsFormat {
    Bits raw = new Bits.MatchAllBits(2);
    int reads, writes, listings;
    boolean fail;
    @Override public Bits readLiveDocs(Directory dir, SegmentCommitInfo info, IOContext context) throws IOException {
      reads++;
      if (fail) throw new IOException("delegate read");
      return raw;
    }
    @Override public void writeLiveDocs(Bits bits, Directory dir, SegmentCommitInfo info,
        int count, IOContext context) throws IOException {
      writes++;
      if (fail) throw new IOException("delegate write");
    }
    @Override public void files(SegmentCommitInfo info, Collection<String> files) throws IOException {
      listings++;
      files.add("delegate-file");
      if (fail) throw new IOException("delegate files");
    }
  }
  static AssertingLiveDocsFormat format(Probe probe) throws Exception {
    AssertingLiveDocsFormat format = new AssertingLiveDocsFormat();
    var field = AssertingLiveDocsFormat.class.getDeclaredField("in");
    field.setAccessible(true);
    field.set(format, probe);
    return format;
  }
  static class CountProbe implements LiveDocs {
    final int count;
    CountProbe(int count) { this.count = count; }
    @Override public boolean get(int i) { return true; }
    @Override public int length() { return 2; }
    @Override public int deletedCount() { return count; }
    @Override public DocIdSetIterator liveDocsIterator() { return DocIdSetIterator.all(2); }
    @Override public DocIdSetIterator deletedDocsIterator() { return DocIdSetIterator.empty(); }
  }
  @FunctionalInterface interface Operation { void run() throws Exception; }
  static void negative(String name, Operation operation, Probe probe) throws Exception {
    String kind = "ok", message = "-";
    try { operation.run(); }
    catch (AssertionError error) {
      kind = "assert";
      if (error.getMessage() != null) message = error.getMessage();
    } catch (IOException error) {
      kind = "io";
      message = error.getMessage();
    }
    emit("negative", name, kind, message, probe == null ? "-" : probe.reads,
        probe == null ? "-" : probe.writes, probe == null ? "-" : probe.listings);
  }
  static void iterator(String name, String role, LiveDocs bits, int length) throws Exception {
    DocIdSetIterator iterator = role.equals("live") ? bits.liveDocsIterator() : bits.deletedDocsIterator();
    if (iterator.docID() != -1) throw new AssertionError("initial doc");
    StringBuilder docs = new StringBuilder();
    for (int doc = iterator.nextDoc();; doc = iterator.nextDoc()) {
      if (!docs.isEmpty()) docs.append(',');
      docs.append(doc);
      if (iterator.docID() != doc) throw new AssertionError("docID");
      if (doc == DocIdSetIterator.NO_MORE_DOCS) break;
    }
    StringBuilder advances = new StringBuilder();
    for (int target : new int[] {0, 1, 63, 64, 65, 129, Math.max(0, length - 1), length, length + 1}) {
      DocIdSetIterator it = role.equals("live") ? bits.liveDocsIterator() : bits.deletedDocsIterator();
      int doc = it.advance(target);
      int next = doc == DocIdSetIterator.NO_MORE_DOCS ? doc : it.nextDoc();
      if (!advances.isEmpty()) advances.append(',');
      advances.append(target).append(':').append(doc).append(':').append(next);
    }
    emit("iterator", name, role, iterator.cost(), docs, advances);
  }
  static void mask(String name, Bits bits, int offset, int length, boolean sparseMask) {
    FixedBitSet mask = new FixedBitSet(length);
    for (int i = 0; i < length; i++) {
      if ((!sparseMask || i % 3 == 0) && i + offset < bits.length()) mask.set(i);
    }
    String before = bits(mask);
    String status = "ok";
    try { bits.applyMask(mask, offset); }
    catch (AssertionError error) {
      if (length != 0) throw error;
      status = "assert";
    }
    emit("mask", name, offset, before, status, bits(mask));
  }
  static void maskBounds(String name, Bits bits, int offset, int length) {
    FixedBitSet mask = new FixedBitSet(length);
    mask.set(0, length);
    String before = bits(mask);
    try {
      bits.applyMask(mask, offset);
      throw new AssertionError("out-of-range mask accepted");
    } catch (AssertionError expected) {
      if (expected.getMessage() != null && expected.getMessage().equals("out-of-range mask accepted")) throw expected;
    }
    emit("mask", name, offset, before, "assert", bits(mask));
  }
  static void valid(int length, int mode, long generation) throws Exception {
    String name = length + "_" + mode;
    FixedBitSet source = new FixedBitSet(length);
    int deleted = 0;
    for (int i = 0; i < length; i++) {
      boolean live = switch (mode) {
        case 0 -> true;
        case 1 -> false;
        case 2 -> i % 3 != 0;
        default -> i != length / 2;
      };
      if (live) source.set(i); else deleted++;
    }
    try (Directory dir = new ByteBuffersDirectory()) {
      AssertingLiveDocsFormat format = new AssertingLiveDocsFormat();
      SegmentCommitInfo before = info(dir, length, generation == 1 ? -1 : generation - 1, 0);
      format.writeLiveDocs(source, dir, before, deleted, IOContext.DEFAULT);
      SegmentCommitInfo after = info(dir, length, generation, deleted);
      Bits read = format.readLiveDocs(dir, after, IOContext.DEFAULT);
      if (!(read instanceof LiveDocs liveDocs)) throw new AssertionError("missing LiveDocs wrapper");
      if (!read.getClass().getSimpleName().equals("AssertingLiveDocs")) throw new AssertionError("wrapper bypassed");
      ArrayList<String> files = new ArrayList<>();
      files.add("existing");
      format.files(after, files);
      if (files.size() != 2 || !files.getFirst().equals("existing")) throw new AssertionError("file delegation");
      byte[] bytes;
      try (var input = dir.openInput(files.get(1), IOContext.DEFAULT)) {
        bytes = new byte[Math.toIntExact(input.length())];
        input.readBytes(bytes, 0, bytes.length);
      }
      emit("valid", name, length, generation, deleted, bits(read), files.get(1), HexFormat.of().formatHex(bytes));
      if (liveDocs.deletedCount() != deleted) throw new AssertionError("deleted count");
      iterator(name, "live", liveDocs, length);
      iterator(name, "deleted", liveDocs, length);
      mask(name, read, 0, length + 8, false); // Trailing clear mask bits must not query past length.
      mask(name, read, length / 2, Math.min(17, length - length / 2), true);
      mask(name, read, length, 0, false);
      maskBounds(name, read, 0, length + 1); // Earlier clearings survive the bounds assertion.
      maskBounds(name, read, length + 1, 1); // A set mask bit beyond the source fails.
      // Corrupt the actual Java file, preserving asserting read's delegate error.
      bytes[bytes.length - 1] ^= 1;
      dir.deleteFile(files.get(1));
      try (var output = dir.createOutput(files.get(1), IOContext.DEFAULT)) { output.writeBytes(bytes, bytes.length); }
      try { format.readLiveDocs(dir, after, IOContext.DEFAULT); throw new AssertionError("corruption accepted"); }
      catch (IOException expected) { }
    }
  }
  public static void main(String[] args) throws Exception {
    if (!GenAssertingLiveDocs.class.desiredAssertionStatus()) throw new AssertionError("run with -ea");
    long generation = 1;
    for (int length : new int[] {0, 1, 63, 64, 65, 130, 4096}) {
      for (int mode = 0; mode < 4; mode++) valid(length, mode, generation++);
    }
    Bits plain = wrapBits(new Bits.MatchAllBits(2));
    negative("bits_negative_index", () -> plain.get(-1), null);
    negative("bits_upper_bound", () -> plain.get(2), null);
    negative("bits_negative_offset", () -> plain.applyMask(new FixedBitSet(1), -1), null);
    negative("bits_past_end_mask_delegate", () -> {
      FixedBitSet mask = new FixedBitSet(1); mask.set(0); plain.applyMask(mask, 10);
      if (!mask.get(0)) throw new AssertionError("raw mask was not delegated");
    }, null);
    negative("live_negative_count", () -> wrapLiveDocs(new CountProbe(-1)).deletedCount(), null);
    negative("live_count_past_length", () -> wrapLiveDocs(new CountProbe(3)).deletedCount(), null);
    try (Directory dir = new ByteBuffersDirectory()) {
      Probe probe = new Probe();
      AssertingLiveDocsFormat format = format(probe);
      negative("format_write_length", () -> format.writeLiveDocs(new Bits.MatchAllBits(1), dir, info(dir, 2, -1, 0), 0, IOContext.DEFAULT), probe);
      negative("format_write_count", () -> format.writeLiveDocs(new Bits.MatchAllBits(2), dir, info(dir, 2, -1, 0), 1, IOContext.DEFAULT), probe);
      probe.raw = new Bits.MatchAllBits(1);
      negative("format_read_length", () -> format.readLiveDocs(dir, info(dir, 2, 1, 0), IOContext.DEFAULT), probe);
      probe.raw = new Bits.MatchAllBits(2);
      negative("format_read_count", () -> format.readLiveDocs(dir, info(dir, 2, 1, 1), IOContext.DEFAULT), probe);
      probe.raw = null;
      negative("format_read_null", () -> format.readLiveDocs(dir, info(dir, 2, 1, 0), IOContext.DEFAULT), probe);
      probe.fail = true;
      negative("format_write_delegate_io", () -> format.writeLiveDocs(new Bits.MatchAllBits(2), dir, info(dir, 2, -1, 0), 0, IOContext.DEFAULT), probe);
      negative("format_read_delegate_io", () -> format.readLiveDocs(dir, info(dir, 2, 1, 0), IOContext.DEFAULT), probe);
      negative("format_files_delegate_io", () -> format.files(info(dir, 2, 1, 0), new ArrayList<>()), probe);
    }
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Files.writeString(out.resolve("asserting_live_docs.tsv"), OUT, StandardCharsets.UTF_8);
  }
}
