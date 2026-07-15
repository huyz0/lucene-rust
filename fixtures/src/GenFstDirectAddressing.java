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
 * {@link FSTCompiler} actually expands into an {@code ARCS_FOR_DIRECT_ADDRESSING}
 * fixed-length-arc node -- the counterpart to {@code GenFstBinarySearch.java}'s
 * fixture, which deliberately spreads its labels far apart to force binary
 * search instead.
 *
 * <p>{@code shouldExpandNodeWithDirectAddressing} (see {@code FSTCompiler.java})
 * picks direct addressing over binary search when the label range is dense
 * enough that the presence-bitset overhead doesn't blow past the (default,
 * no-oversizing) binary-search byte budget -- but only when the range isn't
 * *fully* contiguous: if every label in {@code [firstLabel, lastLabel]} is
 * present (labelRange == numArcs), {@code FSTCompiler} instead always picks
 * {@code ARCS_FOR_CONTINUOUS} (no presence bitset needed at all, see
 * {@code writeNode}'s {@code continuousLabel} check), which this port does
 * not yet support reading. Six single-byte keys with a small alphabet and
 * exactly one gap ('a'..'f', skipping 'g', then 'h') were confirmed (via a
 * scratch probe reading back {@code arc.toString()}, which prints "(da)" for
 * {@code ARCS_FOR_DIRECT_ADDRESSING}) to make the compiler pick direct
 * addressing -- not continuous or binary search -- for the root node with
 * this exact Lucene 10.5.0 build.
 */
public class GenFstDirectAddressing {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("fst_direct_addressing");
    Files.createDirectories(out);

    // Dense but not fully contiguous single-byte labels -- unlike
    // GenFstBinarySearch's widely spread ones -- so the compiler's cost
    // heuristic prefers direct addressing's small presence bitset over
    // binary search's per-arc label byte and larger sparse array, while the
    // one gap ('g' is missing) keeps it from qualifying as the fully
    // contiguous ARCS_FOR_CONTINUOUS case.
    int[] labels = {'a', 'b', 'c', 'd', 'e', 'f', 'h'};
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
    // ARCS_FOR_DIRECT_ADDRESSING (not silently falling back to list-encoding
    // or binary search, which would defeat the point of this fixture).
    FST<BytesRef> reloaded = FST.read(fstFile, outputs);
    FST.Arc<BytesRef> rootArc = reloaded.getFirstArc(new FST.Arc<>());
    FST.BytesReader r = reloaded.getBytesReader();
    reloaded.readFirstRealTargetArc(rootArc.target(), rootArc, r);
    String arcStr = rootArc.toString();
    if (!arcStr.contains("(da)")) {
      throw new AssertionError(
          "expected root node to be expanded as ARCS_FOR_DIRECT_ADDRESSING ('(da)'), got: "
              + arcStr);
    }

    for (var e : entries.entrySet()) {
      BytesRef got = Util.get(reloaded, e.getKey());
      if (got == null || !got.utf8ToString().equals(e.getValue())) {
        throw new AssertionError(
            "self-check failed for key=" + e.getKey() + " got=" + got + " want=" + e.getValue());
      }
    }

    // Absent: 'g' is the one deliberate gap inside the label range (present
    // bit clear, but arc index in range -- exercises the bit-table rejection
    // path specifically, not just the range-bounds check), plus bytes just
    // outside the range and a couple of clearly disjoint values.
    List<Integer> absentLabels = List.of((int) '`', (int) 'g', (int) 'i', (int) 'A', 0, 255);
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

    System.out.println("FST (ARCS_FOR_DIRECT_ADDRESSING) fixture written to " + out);
  }

  private static String hex(BytesRef b) {
    return HexFormat.of().formatHex(
        java.util.Arrays.copyOfRange(b.bytes, b.offset, b.offset + b.length));
  }
}
