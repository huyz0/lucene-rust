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
 * Generates a real on-heap {@code FST<BytesRef>} where the array-encoded
 * (fixed-length-arc) nodes sit one level *below* the root, not at the root
 * itself -- unlike {@code GenFstBinarySearch.java}/{@code GenFstDirectAddressing.java}/
 * {@code GenFstContinuous.java}, whose single-byte keys always put the array
 * node at the root.
 *
 * <p>This matters for {@code seek_ceil}/{@code seek_floor}: the backtracking
 * helpers that operate on a non-root, non-final arc that itself targets an
 * array-encoded node (e.g. {@code read_last_target_arc}'s array branch,
 * {@code find_next_floor_arc_binary_search}/{@code _direct_addressing}/
 * {@code _continuous}) are only reachable when a seek must recurse *past* the
 * top level -- every prior fixture's root-only array node never exercises
 * those paths.
 *
 * <p>Each two-byte key is `<common-prefix><label>`, where each common-prefix
 * byte ('B', 'D', 'C') groups a label set known (from the sibling fixtures
 * above) to make {@link FSTCompiler} pick {@code ARCS_FOR_BINARY_SEARCH},
 * {@code ARCS_FOR_DIRECT_ADDRESSING}, and {@code ARCS_FOR_CONTINUOUS}
 * respectively for the depth-1 node under that prefix. The root itself stays
 * list-encoded (only 3 arcs: 'B', 'C', 'D' -- far below the 5-arc expansion
 * threshold at shallow depth), so every array node in this fixture is
 * genuinely non-root.
 */
public class GenFstSeekNonRootArrayNode {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("fst_seek_non_root_array_node");
    Files.createDirectories(out);

    int[] binarySearchLabels = {1, 40, 80, 120, 160, 200, 240};
    int[] directAddressingLabels = {'a', 'b', 'c', 'd', 'e', 'f', 'h'}; // gap at 'g'
    int[] continuousLabels = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}; // fully contiguous

    TreeMap<BytesRef, String> entries = new TreeMap<>();
    addPrefixed(entries, (byte) 'B', binarySearchLabels, "bs");
    addPrefixed(entries, (byte) 'D', directAddressingLabels, "da");
    addPrefixed(entries, (byte) 'C', continuousLabels, "cs");

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

    // Self-check: confirm the root stayed list-encoded (so every array node
    // below really is non-root) and that each depth-1 node under 'B'/'C'/'D'
    // was expanded into the expected encoding.
    FST<BytesRef> reloaded = FST.read(fstFile, outputs);
    FST.BytesReader r = reloaded.getBytesReader();

    FST.Arc<BytesRef> root = reloaded.getFirstArc(new FST.Arc<>());
    if (root.bytesPerArc() != 0) {
      throw new AssertionError("expected root to stay list-encoded, got: " + root);
    }

    checkChildEncoding(reloaded, r, (byte) 'B', "(bs)");
    checkChildEncoding(reloaded, r, (byte) 'C', "(cs)");
    checkChildEncoding(reloaded, r, (byte) 'D', "(da)");

    for (var e : entries.entrySet()) {
      BytesRef got = Util.get(reloaded, e.getKey());
      if (got == null || !got.utf8ToString().equals(e.getValue())) {
        throw new AssertionError(
            "self-check failed for key=" + e.getKey() + " got=" + got + " want=" + e.getValue());
      }
    }

    // Absent keys spanning: an in-range gap (direct-addressing 'g'), values
    // just outside each prefix's label range, and a wholly disjoint prefix.
    List<byte[]> absentKeys =
        List.of(
            new byte[] {'D', 'g'},
            new byte[] {'B', 0},
            new byte[] {'B', (byte) 255},
            new byte[] {'C', 'h'},
            new byte[] {'A', 'a'});
    for (byte[] k : absentKeys) {
      BytesRef got = Util.get(reloaded, new BytesRef(k));
      if (got != null) {
        throw new AssertionError("expected key=" + new BytesRef(k) + " to be absent, got=" + got);
      }
    }

    StringBuilder m = new StringBuilder();
    m.append("keyCount=").append(entries.size()).append('\n');
    m.append("fstBytes=").append(Files.size(fstFile)).append('\n');
    Files.writeString(out.resolve("manifest.properties"), m.toString());
  }

  private static void addPrefixed(
      TreeMap<BytesRef, String> entries, byte prefix, int[] labels, String tag)
      throws IOException {
    for (int i = 0; i < labels.length; i++) {
      byte[] k = {prefix, (byte) labels[i]};
      entries.put(new BytesRef(k), tag + i);
    }
  }

  private static void checkChildEncoding(
      FST<BytesRef> fst, FST.BytesReader r, byte prefixLabel, String expectedMarker)
      throws IOException {
    FST.Arc<BytesRef> arc = fst.getFirstArc(new FST.Arc<>());
    fst.findTargetArc(prefixLabel & 0xff, arc, arc, r);
    FST.Arc<BytesRef> child = new FST.Arc<>();
    fst.readFirstRealTargetArc(arc.target(), child, r);
    String arcStr = child.toString();
    if (!arcStr.contains(expectedMarker)) {
      throw new AssertionError(
          "expected node under prefix '"
              + (char) prefixLabel
              + "' to be expanded as "
              + expectedMarker
              + ", got: "
              + arcStr);
    }
  }
}
