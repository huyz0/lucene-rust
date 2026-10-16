package org.apache.lucene.search.highlight;

import java.io.IOException;
import java.util.Map;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.search.Query;

/**
 * M13 T13.2: reaches {@code WeightedSpanTermExtractor}'s protected {@code
 * setMaxDocCharsToAnalyze}, which the extractor needs before it reads any token (its default
 * limit is zero characters).
 */
public final class HighlightFixtureAccess {
  private HighlightFixtureAccess() {}

  public static Map<String, WeightedSpanTerm> extract(
      Query query, TokenStream stream, String field, int maxChars) throws IOException {
    WeightedSpanTermExtractor ex =
        new WeightedSpanTermExtractor() {
          {
            setMaxDocCharsToAnalyze(maxChars);
          }
        };
    ex.setExpandMultiTermQuery(true);
    return ex.getWeightedSpanTerms(query, 1f, stream, field);
  }
}
