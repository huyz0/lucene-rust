package org.apache.lucene.analysis.icu;

import com.ibm.icu.text.Normalizer2;
import java.io.Reader;

/**
 * GenAnalysisIcu's view of analysis-icu's package-private constructors (compiled with the
 * generators, on the class path, where the split package is allowed -- as {@code
 * SmartcnAccess}).
 */
public final class IcuAccess {
  private IcuAccess() {}

  /** {@code new ICUNormalizer2CharFilter(in, normalizer, bufferSize)}. */
  public static Reader charFilter(Reader in, Normalizer2 normalizer, int bufferSize) {
    return new ICUNormalizer2CharFilter(in, normalizer, bufferSize);
  }
}
