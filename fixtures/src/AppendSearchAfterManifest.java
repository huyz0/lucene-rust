import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Cross-engine ground truth for {@code IndexSearcher.searchAfter(after, query, n)} --
 * how a paginating caller gets page 2 -- appended to two already-checked-in fixture
 * indexes' {@code manifest.properties} <b>without regenerating either of them</b>
 * (same technique, and same reason, as {@link AppendScoringManifest}).
 *
 * <p>Two indexes, because the two things that can go wrong live in different places:
 *
 * <ul>
 *   <li>{@code blocktree_index} -- one segment, so this is purely the ranking test
 *       {@code TopScoreDocCollector} applies: a hit is dropped iff it <em>outranks
 *       or equals</em> {@code after} under {@code HitQueue}'s order (higher score
 *       first, and on a score tie the <b>lower</b> doc id first). {@code big}'s
 *       300 documents take four distinct scores, so consecutive pages straddle
 *       score ties and the doc-id half of that rule is actually exercised.
 *   <li>{@code multi_segment_scoring_index} -- two segments, so this is the
 *       {@code afterDoc = after.doc - context.docBase} translation. {@code body:dog}
 *       matches in both segments and interleaves them in the ranking, so a page
 *       boundary falls inside the second segment while the first is still being
 *       filtered against a global doc id it does not use.
 * </ul>
 *
 * <p>Each key records three consecutive pages, so a port cannot pass by getting
 * only the first boundary right.
 *
 * <p>Idempotent: re-running replaces any previously-appended {@code after.*} lines
 * rather than duplicating them.
 */
public class AppendSearchAfterManifest {

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]);
    pages(root.resolve("blocktree_index"), "after.big", new TermQuery(new Term("big", "everywhere")), 5);
    pages(
        root.resolve("multi_segment_scoring_index"),
        "after.dog",
        new TermQuery(new Term("body", "dog")),
        2);
  }

  /** Records {@code <key>.page1}..{@code .page3}: the first page and two {@code searchAfter}s. */
  private static void pages(Path indexDir, String key, Query query, int pageSize)
      throws IOException {
    Path manifestPath = indexDir.resolve("manifest.properties");
    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      out.append(key).append(".query=").append(query).append('\n');
      out.append(key).append(".pageSize=").append(pageSize).append('\n');

      TopDocs page = searcher.search(query, pageSize);
      record(out, key + ".page1", page);
      for (int n = 2; n <= 3; n++) {
        ScoreDoc after =
            page.scoreDocs.length == 0 ? null : page.scoreDocs[page.scoreDocs.length - 1];
        if (after == null) {
          record(out, key + ".page" + n, null);
          continue;
        }
        page = searcher.searchAfter(after, query, pageSize);
        record(out, key + ".page" + n, page);
      }
    }

    String existing = Files.readString(manifestPath);
    StringBuilder kept = new StringBuilder();
    for (String line : existing.split("\n", -1)) {
      if (line.startsWith(key + ".")) {
        continue;
      }
      kept.append(line).append('\n');
    }
    String base = kept.toString();
    while (base.endsWith("\n\n")) {
      base = base.substring(0, base.length() - 1);
    }
    Files.writeString(manifestPath, base + out);
    System.out.println("appended " + key + ".* ground truth to " + manifestPath);
  }

  /**
   * {@code <key>.docScores} as decimal and {@code <key>.bits} as raw
   * {@code Float.floatToIntBits}, so the Rust side compares exact float bits.
   */
  private static void record(StringBuilder out, String key, TopDocs td) {
    StringBuilder scores = new StringBuilder();
    StringBuilder bits = new StringBuilder();
    if (td != null) {
      for (ScoreDoc sd : td.scoreDocs) {
        if (scores.length() > 0) {
          scores.append(',');
          bits.append(',');
        }
        scores.append(sd.doc).append(':').append(sd.score);
        bits.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
      }
    }
    out.append(key).append(".docScores=").append(scores).append('\n');
    out.append(key).append(".bits=").append(bits).append('\n');
  }
}
