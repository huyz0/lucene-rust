import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.CommonTermsQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * {@code IndexSearcher.search(query, 10)}'s lower-bound total -- the count {@code
 * TopScoreDocCollector} reaches under its default threshold of 1000 hits, with the relation --
 * for required-plus-optional shapes, appended to {@code mixed_boolean_scoring_index}'s {@code
 * manifest.properties} <b>without regenerating the index</b> (same technique, and same reason, as
 * {@link AppendScoringManifest}).
 *
 * <p>Once the collector publishes a minimum competitive score, {@code ReqOptSumScorer} (which
 * {@code BooleanScorerSupplier} runs under {@code DefaultBulkScorer} for {@code MUST} + {@code
 * SHOULD} with no minimum) skips blocks of its required clause whose maxima cannot compete and
 * hands every other required match to the collector, which counts it. So the
 * {@code GreaterThanOrEqualTo} total is a function of that scorer's exact iteration, and an
 * engine that drops non-competitive documents one at a time reports a lower count with the same
 * top hits. The shapes: a {@code CommonTermsQuery} that splits into one rare and two frequent
 * terms (rewritten to {@code +rare (f1 f2)}, then flattened to {@code +rare f1 f2}), that
 * flattened boolean written out, one optional clause, two required clauses, a filter as the
 * required clause, and a {@code MUST_NOT} around the shape.
 *
 * <p>Keys: {@code searcher_total.N.query} (the grammar of {@link GenMixedBooleanScoring}, or
 * {@code ct MAX_FREQ TERM...} for a {@code CommonTermsQuery} on {@code body} with both occurs
 * {@code SHOULD}), {@code .hits} ({@code doc:scoreBits}), {@code .total}, {@code .relation}.
 *
 * <p>Idempotent: re-running replaces any previously-appended {@code searcher_total.*} lines.
 */
public class AppendSearcherTotalsManifest {
  static final String PREFIX = "searcher_total.";

  static final String[] QUERIES = {
    "ct 0.5 w5 w0 w1",
    "(b 0 (+ (t w5)) (? (t w0)) (? (t w1)))",
    "(b 0 (+ (t w2)) (? (t w0)) (? (t w1)) (? (t w3)))",
    "(b 0 (+ (t w3)) (? (t w0)))",
    "(b 0 (+ (t w2)) (+ (t w4)) (? (t w0)) (? (t w1)))",
    "(b 0 (# (t w2)) (? (t w0)) (? (t w1)))",
    "(b 0 (+ (t w2)) (? (t w0)) (? (t w1)) (- (t w9)))",
  };

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("mixed_boolean_scoring_index");
    Path manifestPath = indexDir.resolve("manifest.properties");
    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      out.append(PREFIX).append("count=").append(QUERIES.length).append('\n');
      for (int i = 0; i < QUERIES.length; i++) {
        String k = PREFIX + i;
        TopDocs td = searcher.search(parse(QUERIES[i]), 10);
        if (td.totalHits.relation() != TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO) {
          throw new AssertionError("expected a lower-bound total for " + QUERIES[i]);
        }
        StringBuilder hits = new StringBuilder();
        for (ScoreDoc sd : td.scoreDocs) {
          if (hits.length() > 0) {
            hits.append(',');
          }
          hits.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
        }
        out.append(k).append(".query=").append(QUERIES[i]).append('\n');
        out.append(k).append(".hits=").append(hits).append('\n');
        out.append(k).append(".total=").append(td.totalHits.value()).append('\n');
        out.append(k).append(".relation=gte\n");
      }
    }

    String existing = Files.readString(manifestPath);
    StringBuilder kept = new StringBuilder();
    for (String line : existing.split("\n", -1)) {
      if (line.startsWith(PREFIX)) {
        continue;
      }
      kept.append(line).append('\n');
    }
    String base = kept.toString();
    while (base.endsWith("\n\n")) {
      base = base.substring(0, base.length() - 1);
    }
    Files.writeString(manifestPath, base + out);
    System.out.println("appended " + PREFIX + "* ground truth to " + manifestPath);
  }

  static Query parse(String s) {
    if (s.startsWith("ct ")) {
      String[] parts = s.split(" ");
      CommonTermsQuery q =
          new CommonTermsQuery(
              BooleanClause.Occur.SHOULD, BooleanClause.Occur.SHOULD, Float.parseFloat(parts[1]));
      for (int i = 2; i < parts.length; i++) {
        q.add(new Term("body", parts[i]));
      }
      return q;
    }
    return GenMixedBooleanScoring.parse(new GenMixedBooleanScoring.Tokens(s));
  }
}
