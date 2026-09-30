package org.apache.lucene.index;

import java.io.IOException;
import java.util.List;

/**
 * Test-only door into {@code org.apache.lucene.index}'s package-private {@code
 * SlowCompositeCodecReaderWrapper}, for {@code GenReaderApi}: the class is what {@code
 * addIndexes(CodecReader...)} and {@code OneMerge.reorder} read a merge through, and the fixture
 * records what it reads.
 */
public final class ReaderApiAccess {
  private ReaderApiAccess() {}

  /** {@code SlowCompositeCodecReaderWrapper.wrap(readers)}. */
  public static CodecReader slowComposite(List<CodecReader> readers) throws IOException {
    return SlowCompositeCodecReaderWrapper.wrap(readers);
  }
}
