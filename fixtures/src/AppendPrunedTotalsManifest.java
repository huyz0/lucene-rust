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
 * The counts {@link GenMixedBooleanScoring} records hits for but not totals, appended to {@code
 * mixed_boolean_scoring_index}'s {@code manifest.properties} <b>without regenerating the index</b>
 * (same technique, and same reason, as {@link AppendScoringManifest}).
 *
 * <p>Once {@code TopScoreDocCollector} publishes a minimum competitive score, its total is a lower
 * bound: every document the scorer tree hands it is counted, whether it competes or not, so the
 * value is a function of exactly which documents the block-max paths ({@code WANDScorer}, {@code
 * BlockMaxConjunctionScorer}, {@code ReqOptSumScorer}'s impacts approximation, {@code ImpactsDISI})
 * skip. Top hits cannot show an engine that skips more or less than Lucene; these totals do.
 *
 * <p>For every generator query {@code N}: {@code query.N.pruned.total} (the generator's pruned run,
 * {@code TopScoreDocCollectorManager(TOP_N, THRESHOLD)}, whose relation it already records), and
 * {@code query.N.searcher.total} / {@code query.N.searcher.relation} ({@code eq} or {@code gte})
 * from {@code IndexSearcher.search(query, 10)}, the default threshold of 1000.
 *
 * <p>Idempotent: re-running replaces any previously-appended lines of these keys.
 */
public class AppendPrunedTotalsManifest {
  static final Pattern OWN = Pattern.compile("^query\\.\\d+\\.(pruned\\.total|searcher\\.)");

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("mixed_boolean_scoring_index");
    Path manifestPath = indexDir.resolve("manifest.properties");
    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      for (int i = 0; i < GenMixedBooleanScoring.QUERIES.length; i++) {
        Query q =
            GenMixedBooleanScoring.parse(
                new GenMixedBooleanScoring.Tokens(GenMixedBooleanScoring.QUERIES[i]));
        String k = "query." + i;
        TopDocs pruned =
            searcher.search(
                q,
                new TopScoreDocCollectorManager(
                    GenMixedBooleanScoring.TOP_N, GenMixedBooleanScoring.THRESHOLD));
        out.append(k).append(".pruned.total=").append(pruned.totalHits.value()).append('\n');
        TopDocs top = searcher.search(q, 10);
        out.append(k).append(".searcher.total=").append(top.totalHits.value()).append('\n');
        out.append(k)
            .append(".searcher.relation=")
            .append(top.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte")
            .append('\n');
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
    System.out.println("appended pruned/searcher totals to " + manifestPath);
  }
}
