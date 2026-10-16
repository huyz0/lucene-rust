import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.misc.SweetSpotSimilarity;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M14 T14.3 `misc`: {@link SweetSpotSimilarity}. Writes
 * {@code misc/sweet_spot.tsv}, whose rows are
 *
 * <pre>
 * F  config  function  input  floatBits        a function's value (baselineTf, hyperbolicTf, lengthNorm)
 * S  config  query     rank   docId  floatBits  a hit's score over misc/high_freq (written by GenMisc)
 * T  config  toString
 * </pre>
 *
 * Read by {@code crates/lucene-search/tests/misc_fixtures.rs}.
 */
public class GenMiscSweetSpot {

  private static SweetSpotSimilarity make(String config) {
    SweetSpotSimilarity s = new SweetSpotSimilarity();
    if (config.equals("tuned")) {
      s.setBaselineTfFactors(0.5f, 2.0f);
      s.setHyperbolicTfFactors(0.5f, 3.0f, 1.7d, 6.0f);
      s.setLengthNormFactors(3, 9, 0.7f);
    }
    return s;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("misc");
    float[] freqs = {0f, 0.5f, 1f, 2f, 3f, 5f, 10f, 20f, 50f, 1000f, 10000f};
    int[] lengths = {0, 1, 2, 3, 5, 9, 12, 30, 1000};
    String[] queries = {"alpha", "omega", "alpha beta", "title:gamma"};
    try (PrintWriter tsv =
            new PrintWriter(
                Files.newBufferedWriter(out.resolve("sweet_spot.tsv"), StandardCharsets.UTF_8));
        Directory dir = FSDirectory.open(out.resolve("high_freq"));
        IndexReader reader = DirectoryReader.open(dir)) {
      for (String config : new String[] {"default", "tuned"}) {
        SweetSpotSimilarity s = make(config);
        for (float f : freqs) {
          tsv.printf("F\t%s\tbaselineTf\t%s\t%d%n", config, f, Float.floatToRawIntBits(s.baselineTf(f)));
          tsv.printf("F\t%s\thyperbolicTf\t%s\t%d%n", config, f, Float.floatToRawIntBits(s.hyperbolicTf(f)));
          tsv.printf("F\t%s\ttf\t%s\t%d%n", config, f, Float.floatToRawIntBits(s.tf(f)));
        }
        for (int n : lengths) {
          tsv.printf("F\t%s\tlengthNorm\t%d\t%d%n", config, n, Float.floatToRawIntBits(s.lengthNorm(n)));
        }
        tsv.printf("T\t%s\t%s%n", config, s);
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(s);
        for (String q : queries) {
          ScoreDoc[] hits = searcher.search(query(q), 10).scoreDocs;
          for (int rank = 0; rank < hits.length; rank++) {
            tsv.printf(
                "S\t%s\t%s\t%d\t%d\t%d%n",
                config, q, rank, hits[rank].doc, Float.floatToRawIntBits(hits[rank].score));
          }
        }
      }
    }
  }

  /** {@code field:term} or a conjunction-free disjunction of body terms. */
  private static Query query(String q) {
    BooleanQuery.Builder b = new BooleanQuery.Builder();
    for (String part : q.split(" ")) {
      Term t =
          part.contains(":")
              ? new Term(part.substring(0, part.indexOf(':')), part.substring(part.indexOf(':') + 1))
              : new Term("body", part);
      b.add(new TermQuery(t), BooleanClause.Occur.SHOULD);
    }
    return b.build();
  }
}
