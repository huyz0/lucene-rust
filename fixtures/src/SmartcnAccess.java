package org.apache.lucene.analysis.cn.smart.hhmm;

import java.util.ArrayList;
import java.util.List;

/**
 * GenAnalysisSmartcn's view of smartcn's package-private dictionaries (compiled with the
 * generators, on the class path, where the split package is allowed -- as {@code
 * ReaderApiAccess}).
 */
public final class SmartcnAccess {
  private SmartcnAccess() {}

  public static int frequency(char[] w) {
    return WordDictionary.getInstance().getFrequency(w);
  }

  public static int prefixMatch(char[] w) {
    return WordDictionary.getInstance().getPrefixMatch(w);
  }

  public static boolean isEqual(char[] w, int i) {
    return WordDictionary.getInstance().isEqual(w, i);
  }

  public static int bigramFrequency(char[] p) {
    return BigramDictionary.getInstance().getFrequency(p);
  }

  /** {@code HHMMSegmenter.process}: text, start, end, type, weight per token. */
  public static List<String[]> process(String sentence) {
    List<String[]> out = new ArrayList<>();
    for (SegToken t : new HHMMSegmenter().process(sentence)) {
      out.add(new String[] {new String(t.charArray), "" + t.startOffset, "" + t.endOffset, "" + t.wordType, "" + t.weight});
    }
    return out;
  }
}
