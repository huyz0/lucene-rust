import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Pattern;

/**
 * Filtered disjunctions ({@code FILTER} clauses beside two or more {@code SHOULD}, {@code
 * minimum_should_match} 1) whose filter is sparse, appended to {@code mixed_boolean_scoring_index}'s
 * {@code manifest.properties} <b>without regenerating the index</b> (same technique as {@link
 * AppendPrunedTotalsManifest}).
 *
 * <p>{@code BooleanScorerSupplier.filteredOptionalBulkScorer} hands these to a {@code
 * MaxScoreBulkScorer} with a filter, which turns on {@code filterMatches} -- {@code
 * fillScoreBufferViaBitSet}: the filter's matches in each inner window go into a bit set first and
 * only the essential clauses' documents found in it are scored -- when the filter is no denser
 * than every optional clause, or when there are more than four optional clauses and their summed
 * cost is at least the filter's. The generator's own filtered queries use the dense {@code w0} as
 * the filter, so they all take the leap-frog path instead.
 *
 * <p>For every query {@code N}: {@code filtered_ms.N} (the query, in {@link
 * GenMixedBooleanScoring}'s grammar), {@code filtered_ms.N.pruned} / {@code .pruned.relation} /
 * {@code .pruned.total} ({@code TopScoreDocCollectorManager(TOP_N, THRESHOLD)}), {@code
 * filtered_ms.N.searcher.total} / {@code .searcher.relation} ({@code IndexSearcher.search(query,
 * 10)}), and {@code filtered_ms.N.exact} / {@code .total} (an exact count), plus {@code
 * filtered_ms_count}.
 *
 * <p>Idempotent: re-running replaces any previously-appended {@code filtered_ms} lines.
 */
public class AppendFilteredMaxScoreManifest {
  static final Pattern OWN = Pattern.compile("^filtered_ms[._]");

  static final String[] QUERIES = {
    // A rare filter, sparser than both optional clauses.
    "(b 1 (# (t w20)) (? (t w0)) (? (t w1)))",
    // A very rare filter and three optional clauses.
    "(b 1 (# (t w45)) (? (t w0)) (? (t w1)) (? (t w2)))",
    // A filter denser than some of five optional clauses, whose summed cost is larger.
    "(b 1 (# (t w3)) (? (t w0)) (? (t w1)) (? (t w5)) (? (t w10)) (? (t w30)))",
    // Two filters: their conjunction is the filter.
    "(b 1 (# (t w12)) (# (t w2)) (? (t w0)) (? (t w1)))",
    // A prohibited clause around the filtered disjunction (`ReqExclBulkScorer`).
    "(b 1 (# (t w25)) (? (t w0)) (? (t w1)) (- (t w2)))",
    // Optional clauses that are not all terms.
    "(b 1 (# (t w30)) (? (p w0 w1)) (? (t w1)) (? (boost 2 (t w2))))",
    // A filter as dense as the optional clauses (leap-frog), for contrast.
    "(b 1 (# (t w1)) (? (t w2)) (? (t w3)))",
  };

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("mixed_boolean_scoring_index");
    Path manifestPath = indexDir.resolve("manifest.properties");
    StringBuilder out = new StringBuilder();
    out.append("filtered_ms_count=").append(QUERIES.length).append('\n');
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      for (int i = 0; i < QUERIES.length; i++) {
        Query q = GenMixedBooleanScoring.parse(new GenMixedBooleanScoring.Tokens(QUERIES[i]));
        String k = "filtered_ms." + i;
        out.append(k).append('=').append(QUERIES[i]).append('\n');
        TopDocs pruned =
            searcher.search(
                q,
                new TopScoreDocCollectorManager(
                    GenMixedBooleanScoring.TOP_N, GenMixedBooleanScoring.THRESHOLD));
        out.append(k).append(".pruned=").append(GenMixedBooleanScoring.hits(pruned)).append('\n');
        out.append(k).append(".pruned.relation=").append(relation(pruned)).append('\n');
        out.append(k).append(".pruned.total=").append(pruned.totalHits.value()).append('\n');
        TopDocs top = searcher.search(q, 10);
        out.append(k).append(".searcher.total=").append(top.totalHits.value()).append('\n');
        out.append(k).append(".searcher.relation=").append(relation(top)).append('\n');
        TopDocs exact =
            searcher.search(
                q, new TopScoreDocCollectorManager(GenMixedBooleanScoring.TOP_N, Integer.MAX_VALUE));
        if (exact.totalHits.relation() != TotalHits.Relation.EQUAL_TO) {
          throw new AssertionError("exact count expected for " + QUERIES[i]);
        }
        out.append(k).append(".exact=").append(GenMixedBooleanScoring.hits(exact)).append('\n');
        out.append(k).append(".total=").append(exact.totalHits.value()).append('\n');
      }
    }

    String existing = Files.readString(manifestPath);
    StringBuilder kept = new StringBuilder();
    for (String line : existing.split("\n", -1)) {
      if (OWN.matcher(line).find()) {
        continue;
      }
      kept.append(line).append('\n');
    }
    String base = kept.toString();
    while (base.endsWith("\n\n")) {
      base = base.substring(0, base.length() - 1);
    }
    Files.writeString(manifestPath, base + out);
    System.out.println("appended filtered MaxScore queries to " + manifestPath);
  }

  static String relation(TopDocs td) {
    return td.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte";
  }
}
