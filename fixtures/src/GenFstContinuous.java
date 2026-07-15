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
 * {@link FSTCompiler} actually expands into an {@code ARCS_FOR_CONTINUOUS}
 * fixed-length-arc node -- the counterpart to {@code GenFstDirectAddressing.java}'s
 * fixture, which deliberately leaves one gap in its label range to force direct
 * addressing instead.
 *
 * <p>{@code writeNode} (see {@code FSTCompiler.java}) picks {@code ARCS_FOR_CONTINUOUS}
 * over both binary search and direct addressing whenever a node's label range is
 * *fully* contiguous (every label in {@code [firstLabel, lastLabel]} is present,
 * i.e. {@code labelRange == numArcs}) -- no presence bitset is needed at all in that
 * case, so it always wins the cost comparison once the range qualifies. Seven
 * single-byte keys with a small, fully contiguous alphabet (`a`..`g`, no gaps) were
 * confirmed (via a self-check reading back {@code arc.toString()}, which prints
 * "(cs)" for {@code ARCS_FOR_CONTINUOUS}) to make the compiler pick continuous
 * encoding -- not direct addressing or binary search -- for the root node with this
 * exact Lucene 10.5.0 build.
 */
public class GenFstContinuous {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("fst_continuous");
    Files.createDirectories(out);

    // Fully contiguous single-byte labels -- unlike GenFstDirectAddressing's one
    // deliberate gap -- so the compiler's cost heuristic always prefers continuous
    // encoding (no presence bitset needed) over both binary search and direct
    // addressing.
    int[] labels = {'a', 'b', 'c', 'd', 'e', 'f', 'g'};
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
    // ARCS_FOR_CONTINUOUS (not silently falling back to list-encoding, binary
    // search, or direct addressing, which would defeat the point of this fixture).
    FST<BytesRef> reloaded = FST.read(fstFile, outputs);
    FST.Arc<BytesRef> rootArc = reloaded.getFirstArc(new FST.Arc<>());
    FST.BytesReader r = reloaded.getBytesReader();
    reloaded.readFirstRealTargetArc(rootArc.target(), rootArc, r);
    String arcStr = rootArc.toString();
    if (!arcStr.contains("(cs)")) {
      throw new AssertionError(
          "expected root node to be expanded as ARCS_FOR_CONTINUOUS ('(cs)'), got: "
              + arcStr);
    }

    for (var e : entries.entrySet()) {
      BytesRef got = Util.get(reloaded, e.getKey());
      if (got == null || !got.utf8ToString().equals(e.getValue())) {
        throw new AssertionError(
            "self-check failed for key=" + e.getKey() + " got=" + got + " want=" + e.getValue());
      }
    }

    // Absent: bytes strictly outside the contiguous label range, plus a couple of
    // clearly disjoint values. There is no in-range gap to exercise here (unlike
    // direct addressing) since ARCS_FOR_CONTINUOUS's whole point is that every
    // label in its range is present -- the only rejection path is the
    // before/after-range bounds check.
    List<Integer> absentLabels = List.of((int) '`', (int) 'h', (int) 'z', (int) 'A', 0, 255);
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

    System.out.println("FST (ARCS_FOR_CONTINUOUS) fixture written to " + out);
  }

  private static String hex(BytesRef b) {
    return HexFormat.of().formatHex(
        java.util.Arrays.copyOfRange(b.bytes, b.offset, b.offset + b.length));
  }
}
