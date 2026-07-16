package org.apache.lucene.util.bkd;

import java.lang.reflect.Field;
import org.apache.lucene.store.IndexInput;

/**
 * Lives in `org.apache.lucene.util.bkd` (real Lucene's package, not this
 * repo's) purely so it can read {@link BKDReader}'s package-private fields
 * directly -- this is an independent, from-scratch walk of the on-disk BKD
 * index tree and leaf-block header, used to mechanically verify what byte
 * value real Lucene's {@code BKDWriter} actually wrote for `compressedDim`
 * in each leaf of a fixture, without going through (or trusting) this
 * project's Rust decoder at all.
 *
 * <p>The three fields this needs -- {@code indexStartPointer}, {@code
 * numIndexBytes}, {@code indexIn} -- are declared `private` on {@link
 * BKDReader} (not merely package-private), so those three are pulled via
 * reflection; everything else (`config`, `numLeaves`, `version`, the data
 * `in`, and the package-private {@link DocIdsWriter} class/method) is
 * accessed directly because this class shares Lucene's real package.
 *
 * <p>The tree-walk and leaf-header algorithm here is a fresh port of the
 * documented on-disk layout (see `BKDWriter`/`BKDReader` source), not a call
 * into this repo's `points.rs`.
 */
public final class CompressedDimSpy {
  private CompressedDimSpy() {}

  /**
   * Returns the `compressedDim` marker byte written for each leaf of the
   * given reader's tree, in left-to-right (in-order) leaf order.
   */
  public static int[] leafCompressedDims(BKDReader reader) throws Exception {
    Field indexStartPointerField = BKDReader.class.getDeclaredField("indexStartPointer");
    indexStartPointerField.setAccessible(true);
    long indexStartPointer = (long) indexStartPointerField.get(reader);

    Field numIndexBytesField = BKDReader.class.getDeclaredField("numIndexBytes");
    numIndexBytesField.setAccessible(true);
    int numIndexBytes = (int) numIndexBytesField.get(reader);

    Field indexInField = BKDReader.class.getDeclaredField("indexIn");
    indexInField.setAccessible(true);
    IndexInput indexIn = (IndexInput) indexInField.get(reader);

    BKDConfig config = reader.config;
    int numLeaves = reader.numLeaves;
    int version = reader.version;
    IndexInput dataIn = reader.in.clone();

    IndexInput packedIndex = indexIn.slice("packedIndex", indexStartPointer, numIndexBytes);

    // Same recursive descriptor format as `points.rs`'s `walk_node`/
    // `decode_leaf_pointers`: node 1 is the root; a node id >= numLeaves is
    // a leaf. Each internal node encodes one vint (splitDim/prefix/suffix,
    // whose only relevant fact here is how many trailing raw bytes to skip)
    // and, for non-leaf left children, one vint "leftNumBytes" skip hint;
    // the left subtree is fully walked before the right child's file
    // pointer delta (a vlong) is read.
    long rootFp = packedIndex.readVLong();
    java.util.List<Long> leafFps = new java.util.ArrayList<>();
    walk(packedIndex, 1, rootFp, numLeaves, config, leafFps);

    int[] result = new int[leafFps.size()];
    DocIdsWriter docIdsWriter = new DocIdsWriter(config.maxPointsInLeafNode(), version);
    int[] docIdsScratch = new int[config.maxPointsInLeafNode()];
    for (int leafIdx = 0; leafIdx < leafFps.size(); leafIdx++) {
      dataIn.seek(leafFps.get(leafIdx));
      int count = dataIn.readVInt();
      docIdsWriter.readInts(dataIn, count, docIdsScratch);

      int numDims = config.numDims();
      int bytesPerDim = config.bytesPerDim();
      int[] commonPrefixLengths = new int[numDims];
      for (int dim = 0; dim < numDims; dim++) {
        int prefix = dataIn.readVInt();
        commonPrefixLengths[dim] = prefix;
        if (prefix > 0) {
          dataIn.skipBytes(prefix);
        }
      }

      int compressedDim = dataIn.readByte();
      result[leafIdx] = compressedDim;
      // We only need the marker byte itself -- no need to keep parsing the
      // rest of this leaf's body since we advance to the next leaf's own
      // file pointer next iteration.
    }
    return result;
  }

  private static void walk(
      IndexInput in, int nodeId, long fp, int numLeaves, BKDConfig config, java.util.List<Long> leaves)
      throws Exception {
    if (nodeId >= numLeaves) {
      leaves.add(fp);
      return;
    }
    int code = in.readVInt();
    code = code / config.numIndexDims();
    int prefix = code % (1 + config.bytesPerDim());
    int suffix = config.bytesPerDim() - prefix;
    if (suffix > 0) {
      in.skipBytes(suffix - 1);
    }
    int leftChild = nodeId * 2;
    if (leftChild < numLeaves) {
      in.readVInt(); // leftNumBytes skip hint
    }
    walk(in, leftChild, fp, numLeaves, config, leaves);
    long rightDelta = in.readVLong();
    walk(in, nodeId * 2 + 1, fp + rightDelta, numLeaves, config, leaves);
  }
}
