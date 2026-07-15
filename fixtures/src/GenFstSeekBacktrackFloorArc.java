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
import java.util.TreeMap;

/**
 * Generates three real on-heap {@code FST<BytesRef>}s, one per fixed-length-arc
 * encoding, each with a *root* node in that encoding whose children include one
 * more array-encoded (continuous) node one level below it.
 *
 * <p>This is the missing piece {@code GenFstSeekNonRootArrayNode.java} doesn't
 * cover: {@code FstEnum::seek_floor}'s {@code find_next_floor_arc_binary_search}/
 * {@code _direct_addressing}/{@code _continuous} (ported from {@code FSTEnum
 * .findNextFloorArcBinarySearch}/{@code _DirectAddressing}/{@code _Continuous})
 * are only ever reached from {@code backtrack_to_floor_arc}, which re-reads the
 * *parent* node's first arc and finds the floor arc *within the parent* -- so
 * they're only exercised when backtracking lands on a node that is *itself*
 * array-encoded. {@code GenFstSeekNonRootArrayNode.java}'s array nodes are all
 * one level below a list-encoded root, so backtracking from them always takes
 * the list-node linear-scan branch instead.
 *
 * <p>Each fixture's root reuses the exact label sets already confirmed (by the
 * sibling {@code GenFstBinarySearch}/{@code GenFstDirectAddressing}/
 * {@code GenFstContinuous} fixtures) to force that root's own encoding, and
 * additionally gives the label `'d'` (roughly the middle of each 7-arc range) a
 * further two-byte extension `'d'` + `{a..g}` -- fully contiguous, so `d`'s
 * child node always becomes {@code ARCS_FOR_CONTINUOUS} regardless of which
 * fixture this is. Seeking floor for `"d" + 0x00` (a byte below the child's
 * first label `'a'`) must then backtrack from that child node back up to the
 * root and find the floor arc *within the root* -- exercising
 * {@code find_next_floor_arc_binary_search}/{@code _direct_addressing}/
 * {@code _continuous} for whichever encoding that particular root uses.
 */
public class GenFstSeekBacktrackFloorArc {
  public static void main(String[] args) throws IOException {
    Path base = Path.of(args[0]);

    gen(
        base,
        "fst_seek_floor_backtrack_binary_search",
        new int[] {1, 40, 80, 120, 160, 200, 240},
        3, // index of label 120 ('d'-position) within the array above
        "(bs)");
    gen(
        base,
        "fst_seek_floor_backtrack_direct_addressing",
        new int[] {'a', 'b', 'c', 'd', 'e', 'f', 'h'}, // gap at 'g'
        3, // index of 'd'
        "(da)");
    gen(
        base,
        "fst_seek_floor_backtrack_continuous",
        new int[] {'a', 'b', 'c', 'd', 'e', 'f', 'g'}, // fully contiguous
        3, // index of 'd'
        "(cs)");
  }

  private static void gen(
      Path base, String dirName, int[] rootLabels, int extendIdx, String expectedRootMarker)
      throws IOException {
    Path out = base.resolve(dirName);
    Files.createDirectories(out);

    TreeMap<BytesRef, String> entries = new TreeMap<>();
    for (int i = 0; i < rootLabels.length; i++) {
      if (i == extendIdx) {
        // This root label becomes an internal node: extend with a fully
        // contiguous a..g second byte, forcing its own child node to be
        // ARCS_FOR_CONTINUOUS.
        for (int j = 0; j < 7; j++) {
          byte[] k = {(byte) rootLabels[i], (byte) ('a' + j)};
          entries.put(new BytesRef(k), "ext" + i + "_" + j);
        }
      } else {
        byte[] k = {(byte) rootLabels[i]};
        entries.put(new BytesRef(k), "out" + i);
      }
    }

    ByteSequenceOutputs outputs = ByteSequenceOutputs.getSingleton();
    FSTCompiler<BytesRef> fstCompiler =
        new FSTCompiler.Builder<>(FST.INPUT_TYPE.BYTE1, outputs)
            .allowFixedLengthArcs(true)
            .build();

    IntsRefBuilder scratch = new IntsRefBuilder();
    for (var e : entries.entrySet()) {
      IntsRef input = Util.toIntsRef(e.getKey(), scratch);
      fstCompiler.add(input, new BytesRef(e.getValue()));
    }
    FST.FSTMetadata<BytesRef> metadata = fstCompiler.compile();
    FST<BytesRef> fst = FST.fromFSTReader(metadata, fstCompiler.getFSTReader());

    Path fstFile = out.resolve("fst.bin");
    fst.save(fstFile);

    FST<BytesRef> reloaded = FST.read(fstFile, outputs);
    FST.BytesReader r = reloaded.getBytesReader();

    // Self-check: root really did expand into the expected encoding.
    FST.Arc<BytesRef> root = reloaded.getFirstArc(new FST.Arc<>());
    FST.Arc<BytesRef> rootFirstReal = new FST.Arc<>();
    reloaded.readFirstRealTargetArc(root.target(), rootFirstReal, r);
    if (!rootFirstReal.toString().contains(expectedRootMarker)) {
      throw new AssertionError(
          "["
              + dirName
              + "] expected root to be expanded as "
              + expectedRootMarker
              + ", got: "
              + rootFirstReal);
    }

    // Self-check: the extended label's child node really did become continuous.
    byte extendedLabel = (byte) rootLabels[extendIdx];
    FST.Arc<BytesRef> extArc = reloaded.getFirstArc(new FST.Arc<>());
    reloaded.findTargetArc(extendedLabel & 0xff, extArc, extArc, r);
    FST.Arc<BytesRef> extChild = new FST.Arc<>();
    reloaded.readFirstRealTargetArc(extArc.target(), extChild, r);
    if (!extChild.toString().contains("(cs)")) {
      throw new AssertionError(
          "["
              + dirName
              + "] expected extended label's child to be expanded as (cs), got: "
              + extChild);
    }

    for (var e : entries.entrySet()) {
      BytesRef got = Util.get(reloaded, e.getKey());
      if (got == null || !got.utf8ToString().equals(e.getValue())) {
        throw new AssertionError(
            "[" + dirName + "] self-check failed for key=" + e.getKey() + " got=" + got);
      }
    }

    StringBuilder m = new StringBuilder();
    m.append("keyCount=").append(entries.size()).append('\n');
    m.append("extendedLabel=").append((char) extendedLabel).append('\n');
    m.append("fstBytes=").append(Files.size(fstFile)).append('\n');
    Files.writeString(out.resolve("manifest.properties"), m.toString());
  }
}
