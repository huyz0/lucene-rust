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
}
