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
 * Queries whose scoring clauses are not all present in the segment, appended to {@code
 * mixed_boolean_scoring_index}'s {@code manifest.properties} <b>without regenerating the index</b>
 * (same technique as {@link AppendPrunedTotalsManifest}).
 *
 * <p>{@code BooleanScorerSupplier.setTopLevelScoringClause} passes the call through to its scoring
 * clause when {@code subs.get(SHOULD).size() + subs.get(MUST).size() == 1}, and {@code subs} holds
 * only the suppliers that exist in the segment: a {@code TermQuery} on a term the segment lacks has
 * none. So a boolean that is written with two scoring clauses, one of them absent, hands its lone
 * present clause the top-level flag -- which turns a nested conjunction into a {@code
 * BlockMaxConjunctionScorer} and a nested disjunction into a {@code WANDScorer}, and so changes
 * which documents the pruned run counts.
 *
 * <p>For every query {@code N}: {@code present.N} (the query, in {@link GenMixedBooleanScoring}'s
 * grammar), {@code present.N.pruned} / {@code .pruned.relation} / {@code .pruned.total} ({@code
 * TopScoreDocCollectorManager(TOP_N, THRESHOLD)}), {@code present.N.searcher.total} / {@code
 * .searcher.relation} ({@code IndexSearcher.search(query, 10)}), and {@code present.N.exact} /
 * {@code .total} (an exact count), plus {@code present_count}.
 *
 * <p>Idempotent: re-running replaces any previously-appended {@code present} lines.
 */
public class AppendPresentClausesManifest {
  static final Pattern OWN = Pattern.compile("^present[._]");

  static final String[] QUERIES = {
    // A filter and a nested conjunction that is the lone present scoring clause.
    "(b 0 (# (t w0)) (? (b 0 (+ (t w1)) (+ (t w2)))) (? (t zzabsent)))",
    // A required nested disjunction (`MUST` is never flattened) beside an absent `SHOULD`.
    "(b 0 (+ (b 0 (? (t w4)) (? (t w5)))) (# (t w0)) (? (t zzabsent)))",
    // The root is a disjunction of a nested conjunction and an absent term.
    "(b 0 (? (b 0 (+ (t w1)) (+ (t w3)))) (? (t zzabsent)))",
    // Two absent scoring clauses around a present nested disjunction under `MUST`.
    "(b 0 (+ (b 0 (? (t w2)) (? (t w6)))) (? (t zzabsent)) (? (t zzmissing)) (- (t w9)))",
  };

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("mixed_boolean_scoring_index");
    Path manifestPath = indexDir.resolve("manifest.properties");
    StringBuilder out = new StringBuilder();
    out.append("present_count=").append(QUERIES.length).append('\n');
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      for (int i = 0; i < QUERIES.length; i++) {
        Query q = GenMixedBooleanScoring.parse(new GenMixedBooleanScoring.Tokens(QUERIES[i]));
        String k = "present." + i;
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
    System.out.println("appended present-clause queries to " + manifestPath);
  }

  static String relation(TopDocs td) {
    return td.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte";
  }
}
