package org.apache.lucene.monitor;

import org.apache.lucene.search.Query;

/**
 * Package-private access for the fixture generator: {@link QueryAnalyzer} is package-private in
 * Lucene, so the generator reaches it from here, in the same package.
 */
public final class MonitorFixtureAccess {
  private MonitorFixtureAccess() {}

  public static QueryTree buildTree(Query query, TermWeightor weightor) {
    return new QueryAnalyzer().buildTree(query, weightor);
  }
}
