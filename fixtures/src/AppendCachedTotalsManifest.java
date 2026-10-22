import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LRUQueryCache;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermInSetQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.UsageTrackingQueryCachingPolicy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Totals of a required term set beside one optional term, searched repeatedly through a real
 * {@code LRUQueryCache}, appended to {@code mixed_boolean_scoring_index}'s {@code
 * manifest.properties} <b>without regenerating the index</b> (same technique, and same reason, as
 * {@link AppendScoringManifest}).
 *
 * <p>Each query runs {@link #RUNS} times on a fresh searcher with its own {@code
 * LRUQueryCache(1000, 32 MB)} (both segments are over its 10,000-document floor) and {@code
 * UsageTrackingQueryCachingPolicy}: the first runs iterate the term set's postings, and once the
 * policy has seen it often enough its matches are cached as a bit set and every later run reads
 * them from there. Both run under {@code ReqOptSumScorer}, whose impacts approximation takes its
 * block bounds from the optional term alone (a constant-scoring required clause answers {@code
 * advanceShallow} with {@code NO_MORE_DOCS}), so the lower-bound total is the same for every run:
 * recording each one shows that, and an engine with a separate cached path has to match it.
 *
 * <p>Keys: {@code cached_total.count}, {@code cached_total.runs}, then per query {@code N}: {@code
 * .set} (terms on {@code body}, space-separated), {@code .opt}, {@code .not} (empty for none),
 * {@code .top_n}, {@code .threshold} ({@code Integer.MAX_VALUE}: an exact count), {@code .hits} (the last run's {@code doc:scoreBits}), {@code
 * .totals} ({@code value:eq|gte} per run, comma-separated) and {@code .cached} (whether the cache
 * held an entry after the runs).
 *
 * <p>Idempotent: re-running replaces any previously-appended {@code cached_total.*} lines.
 */
public class AppendCachedTotalsManifest {
  static final String PREFIX = "cached_total.";
  static final int RUNS = 6;

  static final String[][] SHAPES = {
    {"w3 w5 w7", "w0", ""},
    {"w1 w2 w9", "w4", ""},
    {"w3 w5 w7", "w0", "w8"},
    {"w1 w2 w9", "w4", "w6"},
  };

  static final int[][] LIMITS = {
    {10, 10}, {3, 0}, {5, 50}, {10, 1000}, {1, 1}, {10, Integer.MAX_VALUE}
  };

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("mixed_boolean_scoring_index");
    Path manifestPath = indexDir.resolve("manifest.properties");
    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      out.append(PREFIX).append("count=").append(SHAPES.length * LIMITS.length).append('\n');
      out.append(PREFIX).append("runs=").append(RUNS).append('\n');
      int n = 0;
      for (String[] shape : SHAPES) {
        for (int[] limit : LIMITS) {
          IndexSearcher searcher = new IndexSearcher(reader);
          LRUQueryCache cache = new LRUQueryCache(1000, 32L << 20);
          searcher.setQueryCache(cache);
          searcher.setQueryCachingPolicy(new UsageTrackingQueryCachingPolicy());
          Query q = query(shape);
          StringBuilder totals = new StringBuilder();
          TopDocs last = null;
          for (int r = 0; r < RUNS; r++) {
            last = searcher.search(q, new TopScoreDocCollectorManager(limit[0], limit[1]));
            if (r > 0) {
              totals.append(',');
            }
            totals
                .append(last.totalHits.value())
                .append(':')
                .append(last.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte");
          }
          StringBuilder hits = new StringBuilder();
          for (ScoreDoc sd : last.scoreDocs) {
            if (hits.length() > 0) {
              hits.append(',');
            }
            hits.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
          }
          String k = PREFIX + n++;
          out.append(k).append(".set=").append(shape[0]).append('\n');
          out.append(k).append(".opt=").append(shape[1]).append('\n');
          out.append(k).append(".not=").append(shape[2]).append('\n');
          out.append(k).append(".top_n=").append(limit[0]).append('\n');
          out.append(k).append(".threshold=").append(limit[1]).append('\n');
          out.append(k).append(".hits=").append(hits).append('\n');
          out.append(k).append(".totals=").append(totals).append('\n');
          out.append(k).append(".cached=").append(cache.getCacheCount() > 0).append('\n');
        }
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

  static Query query(String[] shape) {
    List<BytesRef> terms = new ArrayList<>();
    for (String t : shape[0].split(" ")) {
      terms.add(new BytesRef(t));
    }
    BooleanQuery.Builder b = new BooleanQuery.Builder();
    b.add(new TermInSetQuery("body", terms), BooleanClause.Occur.MUST);
    b.add(new TermQuery(new Term("body", shape[1])), BooleanClause.Occur.SHOULD);
    if (!shape[2].isEmpty()) {
      b.add(new TermQuery(new Term("body", shape[2])), BooleanClause.Occur.MUST_NOT);
    }
    return b.build();
  }
}
