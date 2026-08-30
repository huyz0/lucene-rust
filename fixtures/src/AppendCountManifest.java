import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Cross-engine ground truth for {@code Weight.count(LeafReaderContext)} and
 * {@link FieldExistsQuery}'s {@code count}/{@code rewrite} shortcuts, appended to
 * five already-checked-in fixture indexes' {@code manifest.properties} <b>without
 * regenerating any of them</b> -- same technique, and same reason, as
 * {@link AppendScoringManifest} (which see: re-running the generators would stamp a
 * fresh random segment id into indexes whose id other suites hardcode).
 *
 * <p>What each index contributes, and why it is the one that can:
 *
 * <ul>
 *   <li>{@code blocktree_index} -- no deletions, so {@code TermQuery}'s count is
 *       {@code termsEnum.docFreq()} straight off the terms dictionary, and
 *       {@code MatchAllDocsQuery}'s is {@code numDocs() == maxDoc()}. Its
 *       {@code body} field indexes norms on only 4 of its documents, which is
 *       {@code FieldExistsQuery}'s norms branch declining to shortcut.
 *   <li>{@code live_docs_index} -- the only committed index with a {@code .liv}
 *       file. With deletions, {@code TermQuery.count} must fall back to a scan:
 *       {@code id:1} names a deleted document and so counts 0 even though its
 *       {@code docFreq} is 1. That single number is what separates a correct
 *       shortcut from one that reports deleted documents as hits.
 *   <li>{@code norms_index} -- {@code body} carries norms on every document
 *       ({@code docCount == maxDoc}), so {@code rewrite} collapses the whole query
 *       to {@code MatchAllDocsQuery}; {@code sparse_body} carries them on 3 of 5
 *       and so does not.
 *   <li>{@code doc_values_index} -- doc-values fields with no terms, no points and
 *       no skip index: nothing gives {@code count} a doc count to work from, so
 *       even the field present on every document has to be scanned. This is the
 *       {@code count == -1} arm, which is easy to get wrong by assuming a dense
 *       doc-values field is always shortcut-able.
 *   <li>{@code doc_values_skip_index} -- a doc-values field with a skip index over
 *       36 000 documents, i.e. the {@code DocValuesSkipper.docCount()} arm, and the
 *       only committed fixture that has one.
 * </ul>
 *
 * <p>Idempotent: re-running replaces any previously-appended {@code count.*} /
 * {@code rewrite.*} lines rather than duplicating them.
 */
public class AppendCountManifest {

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]);

    record(
        root.resolve("blocktree_index"),
        (searcher, out) -> {
          count(out, "count.term.body.cat", searcher, new TermQuery(new Term("body", "cat")));
          count(
              out,
              "count.term.big.everywhere",
              searcher,
              new TermQuery(new Term("big", "everywhere")));
          count(
              out,
              "count.term.absent",
              searcher,
              new TermQuery(new Term("body", "zzz-no-such-term")));
          count(
              out,
              "count.term.absentfield",
              searcher,
              new TermQuery(new Term("no-such-field", "cat")));
          count(out, "count.matchall", searcher, new MatchAllDocsQuery());
          fieldExists(out, "body", searcher);
          out.append("count.maxDoc=").append(searcher.getIndexReader().maxDoc()).append('\n');
          out.append("count.numDocs=").append(searcher.getIndexReader().numDocs()).append('\n');
        });

    record(
        root.resolve("live_docs_index"),
        (searcher, out) -> {
          // docFreq 1, but the document is deleted: the shortcut must not fire.
          count(out, "count.term.id.1", searcher, new TermQuery(new Term("id", "1")));
          count(out, "count.term.id.0", searcher, new TermQuery(new Term("id", "0")));
          count(out, "count.matchall", searcher, new MatchAllDocsQuery());
          out.append("count.maxDoc=").append(searcher.getIndexReader().maxDoc()).append('\n');
          out.append("count.numDocs=").append(searcher.getIndexReader().numDocs()).append('\n');
        });

    record(
        root.resolve("norms_index"),
        (searcher, out) -> {
          fieldExists(out, "body", searcher);
          fieldExists(out, "sparse_body", searcher);
          out.append("count.maxDoc=").append(searcher.getIndexReader().maxDoc()).append('\n');
          out.append("count.numDocs=").append(searcher.getIndexReader().numDocs()).append('\n');
        });

    record(
        root.resolve("doc_values_index"),
        (searcher, out) -> {
          fieldExists(out, "varying", searcher);
          fieldExists(out, "sparse", searcher);
          out.append("count.maxDoc=").append(searcher.getIndexReader().maxDoc()).append('\n');
          out.append("count.numDocs=").append(searcher.getIndexReader().numDocs()).append('\n');
        });

    record(
        root.resolve("doc_values_skip_index"),
        (searcher, out) -> {
          fieldExists(out, "skip_numeric", searcher);
          out.append("count.maxDoc=").append(searcher.getIndexReader().maxDoc()).append('\n');
          out.append("count.numDocs=").append(searcher.getIndexReader().numDocs()).append('\n');
        });
  }

  /** What one index contributes, given an open searcher. */
  private interface Body {
    void run(IndexSearcher searcher, StringBuilder out) throws IOException;
  }

  private static void record(Path indexDir, Body body) throws IOException {
    Path manifestPath = indexDir.resolve("manifest.properties");
    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      body.run(new IndexSearcher(reader), out);
    }

    String existing = Files.readString(manifestPath);
    StringBuilder kept = new StringBuilder();
    for (String line : existing.split("\n", -1)) {
      if (line.startsWith("count.") || line.startsWith("rewrite.")) {
        continue;
      }
      kept.append(line).append('\n');
    }
    String base = kept.toString();
    while (base.endsWith("\n\n")) {
      base = base.substring(0, base.length() - 1);
    }
    Files.writeString(manifestPath, base + out);
    System.out.println("appended count.* ground truth to " + manifestPath);
  }

  /** {@code IndexSearcher.count(query)}, plus the query's own {@code toString}. */
  private static void count(StringBuilder out, String key, IndexSearcher searcher, Query query)
      throws IOException {
    out.append(key).append(".query=").append(query).append('\n');
    out.append(key).append('=').append(searcher.count(query)).append('\n');
  }

  /**
   * A {@link FieldExistsQuery}'s count <b>and</b> what {@code rewrite} makes of it --
   * the two halves have to agree (a query that rewrites to {@code MatchAllDocsQuery}
   * must count {@code numDocs}), and recording only one of them would not show that.
   */
  private static void fieldExists(StringBuilder out, String field, IndexSearcher searcher)
      throws IOException {
    Query q = new FieldExistsQuery(field);
    count(out, "count.fieldexists." + field, searcher, q);
    out.append("rewrite.fieldexists.")
        .append(field)
        .append('=')
        .append(searcher.rewrite(q))
        .append('\n');
  }
}
