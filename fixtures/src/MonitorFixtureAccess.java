package org.apache.lucene.monitor;

import org.apache.lucene.search.Query;
import java.util.List;

/**
 * Package-private access for the fixture generator: {@link QueryAnalyzer} is package-private in
 * Lucene, so the generator reaches it from here, in the same package.
 */
public final class MonitorFixtureAccess {
  private MonitorFixtureAccess() {}

  public static QueryTree buildTree(Query query, TermWeightor weightor) {
    return new QueryAnalyzer().buildTree(query, weightor);
  }

  /**
   * The indexed document of {@code query} under {@code presearcher}: one line per field, its name and
   * its terms in the order the document's token stream yields them.
   */
  public static java.util.List<String> indexedFields(
      Presearcher presearcher, Query query, java.util.Map<String, String> metadata) throws Exception {
    org.apache.lucene.document.Document doc = presearcher.indexQuery(query, metadata);
    java.util.List<String> out = new java.util.ArrayList<>();
    for (org.apache.lucene.index.IndexableField f : doc.getFields()) {
      java.util.List<String> terms = new java.util.ArrayList<>();
      org.apache.lucene.analysis.TokenStream ts = f.tokenStream(null, null);
      if (ts != null) {
        org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute t =
            ts.addAttribute(org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute.class);
        ts.reset();
        while (ts.incrementToken()) {
          terms.add(t.getBytesRef().utf8ToString());
        }
        ts.end();
        ts.close();
      }
      out.add(f.name() + "	" + String.join(" ", terms));
    }
    return out;
  }

  /**
   * {@code presearcher.buildQuery} over a batch of one document: {@code fields} are {@code name=value}
   * pairs, each value analysed by {@code analyzer} into the batch's terms.
   */
  public static String batchQuery(
      Presearcher presearcher, org.apache.lucene.analysis.Analyzer analyzer, String[] fields)
      throws Exception {
    org.apache.lucene.index.memory.MemoryIndex mi = new org.apache.lucene.index.memory.MemoryIndex();
    for (String f : fields) {
      int eq = f.indexOf('=');
      mi.addField(f.substring(0, eq), f.substring(eq + 1), analyzer);
    }
    org.apache.lucene.index.LeafReader reader = (org.apache.lucene.index.LeafReader)
        mi.createSearcher().getIndexReader().leaves().get(0).reader();
    return presearcher.buildQuery(reader, (field, term) -> true).toString();
  }

  /** {@code QueryAnalyzer(handlers).buildTree(query, weightor)}. */
  public static QueryTree buildTreeWith(
      java.util.List<CustomQueryHandler> handlers, Query query, TermWeightor weightor) {
    return new QueryAnalyzer(handlers).buildTree(query, weightor);
  }

  /**
   * {@code SuffixingNGramTokenFilter} over {@code terms}, as one stream: every token it emits, in
   * order.
   */
  public static java.util.List<String> ngrams(
      java.util.List<String> terms, String suffix, String wildcard, int maxTokenLength)
      throws Exception {
    final java.util.Iterator<String> it = terms.iterator();
    org.apache.lucene.analysis.TokenStream in =
        new org.apache.lucene.analysis.TokenStream() {
          final org.apache.lucene.analysis.tokenattributes.CharTermAttribute term =
              addAttribute(org.apache.lucene.analysis.tokenattributes.CharTermAttribute.class);
          final org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute inc =
              addAttribute(org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute.class);
          final org.apache.lucene.analysis.tokenattributes.OffsetAttribute off =
              addAttribute(org.apache.lucene.analysis.tokenattributes.OffsetAttribute.class);

          @Override
          public boolean incrementToken() {
            if (!it.hasNext()) return false;
            clearAttributes();
            String t = it.next();
            term.append(t);
            inc.setPositionIncrement(1);
            off.setOffset(0, t.length());
            return true;
          }
        };
    org.apache.lucene.analysis.TokenStream out =
        new SuffixingNGramTokenFilter(in, suffix, wildcard, maxTokenLength);
    org.apache.lucene.analysis.tokenattributes.CharTermAttribute t =
        out.addAttribute(org.apache.lucene.analysis.tokenattributes.CharTermAttribute.class);
    java.util.List<String> result = new java.util.ArrayList<>();
    out.reset();
    while (out.incrementToken()) {
      result.add(t.toString());
    }
    out.end();
    out.close();
    return result;
  }
}
