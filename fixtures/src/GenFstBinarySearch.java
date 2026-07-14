import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IntsRef;
import org.apache.lucene.util.IntsRefBuilder;
import org.apache.lucene.util.fst.ByteSequenceOutputs;
import org.apache.lucene.util.fst.FST;
import org.apache.lucene.util.fst.FSTCompiler;
import org.apache.lucene.util.fst.Util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;
import java.util.List;
import java.util.TreeMap;

/**
 * Generates a real on-heap {@code FST<BytesRef>} whose root node real Lucene's
 * {@link FSTCompiler} actually expands into an {@code ARCS_FOR_BINARY_SEARCH}
 * fixed-length-arc node -- unlike {@code GenFst.java}'s fixture, which passes
 * {@code allowFixedLengthArcs(false)} specifically to stay in the (smaller)
 * list-encoded-only slice.
 *
 * <p>{@link FSTCompiler} only expands a node into a fixed-length-arc encoding
 * once it has "enough" arcs ({@code FIXED_LENGTH_ARC_SHALLOW_NUM_ARCS} == 5 at
 * shallow depth <= 3, see {@code shouldExpandNodeWithFixedLengthArcs}), and
 * within that, picks {@code ARCS_FOR_BINARY_SEARCH} over
 * {@code ARCS_FOR_DIRECT_ADDRESSING} only when direct addressing's presence
 * bitset would cost more than the oversizing budget allows (see
 * {@code shouldExpandNodeWithDirectAddressing}) -- i.e. when the arcs' labels
 * are sparse relative to their count. Seven single-byte keys with widely
 * spaced byte values (1, 40, 80, 120, 160, 200, 240) were confirmed (via a
 * scratch probe reading back {@code arc.toString()}, which prints "(bs)" for
 * {@code ARCS_FOR_BINARY_SEARCH}) to make the compiler pick binary search for
 * the root node with this exact Lucene 10.5.0 build.
 */
public class GenFstBinarySearch {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("fst_binary_search");
    Files.createDirectories(out);

    // NOTE: keys are raw bytes >= 0x80 (up to 0xf0), so they must be built as
    // `BytesRef(byte[])` directly -- routing them through a `String` first
    // (even with ISO-8859-1) and then `new BytesRef(String)` is wrong: that
    // constructor UTF-8-*encodes* the string, turning any single raw byte
    // >= 0x80 into a *two*-byte UTF-8 sequence and silently changing the key
    // set (this was caught by this fixture's own self-check below failing
    // during development -- the compiler picked "(bs)" but over the wrong,
    // UTF-8-mangled 2-byte keys instead of the intended 7 single-byte keys).
    int[] labels = {1, 40, 80, 120, 160, 200, 240};
    TreeMap<BytesRef, String> entries = new TreeMap<>();
    for (int i = 0; i < labels.length; i++) {
      byte[] k = {(byte) labels[i]};
      entries.put(new BytesRef(k), "out" + i);
    }

    ByteSequenceOutputs outputs = ByteSequenceOutputs.getSingleton();
    FSTCompiler<BytesRef> fstCompiler =
        new FSTCompiler.Builder<>(FST.INPUT_TYPE.BYTE1, outputs)
            .allowFixedLengthArcs(true)
            .build();

    IntsRefBuilder scratch = new IntsRefBuilder();
    for (var e : entries.entrySet()) {
      BytesRef key = e.getKey();
      BytesRef output = new BytesRef(e.getValue());
      IntsRef input = Util.toIntsRef(key, scratch);
      fstCompiler.add(input, output);
    }
    FST.FSTMetadata<BytesRef> metadata = fstCompiler.compile();
    FST<BytesRef> fst = FST.fromFSTReader(metadata, fstCompiler.getFSTReader());

    Path fstFile = out.resolve("fst.bin");
    fst.save(fstFile);

    // Self-check: confirm the root node really did get expanded into
    // ARCS_FOR_BINARY_SEARCH (not silently falling back to list-encoding or
    // direct-addressing, which would defeat the point of this fixture).
    FST<BytesRef> reloaded = FST.read(fstFile, outputs);
    FST.Arc<BytesRef> rootArc = reloaded.getFirstArc(new FST.Arc<>());
    FST.BytesReader r = reloaded.getBytesReader();
    reloaded.readFirstRealTargetArc(rootArc.target(), rootArc, r);
    String arcStr = rootArc.toString();
    if (!arcStr.contains("(bs)")) {
      throw new AssertionError(
          "expected root node to be expanded as ARCS_FOR_BINARY_SEARCH ('(bs)'), got: " + arcStr);
    }

    for (var e : entries.entrySet()) {
      BytesRef got = Util.get(reloaded, e.getKey());
      if (got == null || !got.utf8ToString().equals(e.getValue())) {
        throw new AssertionError(
            "self-check failed for key=" + e.getKey() + " got=" + got + " want=" + e.getValue());
      }
    }

    // Absent: bytes strictly between/outside the chosen labels, plus the
    // empty string (this FST has no empty output).
    List<Integer> absentLabels = List.of(0, 20, 60, 100, 140, 180, 220, 255);
    for (var abLabel : absentLabels) {
      byte[] k = {(byte) (int) abLabel};
      BytesRef got = Util.get(reloaded, new BytesRef(k));
      if (got != null) {
        throw new AssertionError("expected label=" + abLabel + " to be absent, got=" + got);
      }
    }

    StringBuilder m = new StringBuilder();
    m.append("num_present=").append(entries.size()).append('\n');
    int i = 0;
    for (var e : entries.entrySet()) {
      m.append("present.").append(i).append(".key_hex=").append(hex(e.getKey())).append('\n');
      m.append("present.")
          .append(i)
          .append(".output_hex=")
          .append(hex(new BytesRef(e.getValue())))
          .append('\n');
      i++;
    }
    m.append("num_absent=").append(absentLabels.size()).append('\n');
    for (int j = 0; j < absentLabels.size(); j++) {
      byte[] k = {(byte) (int) absentLabels.get(j)};
      m.append("absent.").append(j).append(".key_hex=").append(hex(new BytesRef(k))).append('\n');
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());

    System.out.println("FST (ARCS_FOR_BINARY_SEARCH) fixture written to " + out);
  }

  private static String hex(BytesRef b) {
    return HexFormat.of().formatHex(
        java.util.Arrays.copyOfRange(b.bytes, b.offset, b.offset + b.length));
  }
}
