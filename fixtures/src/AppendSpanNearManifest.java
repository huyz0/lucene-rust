import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Cross-engine ground truth for {@code SpanNearQuery}'s <b>unordered</b> arm,
 * appended to the already-checked-in {@code fixtures/data/blocktree_index/}
 * directory's {@code manifest.properties} <b>without regenerating the index</b>
 * -- same technique, and same reason, as {@link AppendScoringManifest}.
 *
 * <p>{@code GenBlockTree} already records two `inOrder` cases on doc 8558
 * ({@code field.pos.span.*}). What it does not record, and what this port got
 * wrong, is what {@code NearSpansUnordered} does when the chosen sub-spans
 * <b>overlap</b>. Its match test is a pure width test --
 *
 * <pre>{@code
 * boolean atMatch() {
 *   return (maxEndPosition - top().startPosition() - totalSpanLength) <= allowedSlop;
 * }
 * }</pre>
 *
 * <p>-- with no non-overlap rule at all, so two clauses holding the same term
 * may settle on the <em>same</em> position and produce a negative width, which
 * matches. {@code NearSpansOrdered} is the opposite: {@code stretchToOrder}
 * advances each sub-span to {@code >= prevSpans.endPosition()}, so non-overlap
 * is required there. This port rejected overlap on both arms, which drops hits
 * real Lucene returns.
 *
 * <p>Each case records {@code spannear.<name>.docs} -- the matching doc IDs,
 * ascending, comma-separated, empty for none -- plus the query's shape, so a
 * reader can see what was asked without re-deriving it.
 *
 * <p>Idempotent: re-running replaces any previously-appended {@code spannear.*}
 * lines rather than duplicating them.
 */
public class AppendSpanNearManifest {

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("blocktree_index");
    Path manifestPath = indexDir.resolve("manifest.properties");

    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);

      // doc 8555: alpha@0 beta@1   doc 8556: alpha@0 alpha@1
      // doc 8557: alpha@0 beta@3   doc 8558: delta@0 gamma@1

      // The headline: two clauses on one term. Doc 8555 has a single `alpha`,
      // doc 8556 has two. Unordered, one occurrence is enough (width -1);
      // ordered, it is not.
      record(out, "repeat_unordered_slop0", searcher, new String[] {"alpha", "alpha"}, 0, false);
      record(out, "repeat_unordered_slop2", searcher, new String[] {"alpha", "alpha"}, 2, false);
      record(out, "repeat_ordered_slop2", searcher, new String[] {"alpha", "alpha"}, 2, true);
      // A transposition: the unordered width of `beta alpha` over
      // `alpha@0 beta@1` is 0, so slop 0 already matches.
      record(out, "transposed_unordered_slop0", searcher, new String[] {"beta", "alpha"}, 0, false);
      record(out, "transposed_ordered_slop0", searcher, new String[] {"beta", "alpha"}, 0, true);
      // The same terms in phrase order, plus the wider gap in doc 8557.
      record(out, "inorder_slop0", searcher, new String[] {"alpha", "beta"}, 0, true);
      record(out, "inorder_slop2", searcher, new String[] {"alpha", "beta"}, 2, true);
      record(out, "inorder_unordered_slop2", searcher, new String[] {"alpha", "beta"}, 2, false);
      // Three clauses over doc 8556's two `alpha`s: unordered, two of the
      // three must share a position, so this is the multi-clause form of the
      // same question.
      record(
          out,
          "triple_repeat_unordered_slop1",
          searcher,
          new String[] {"alpha", "alpha", "alpha"},
          1,
          false);
    }

    String existing = Files.readString(manifestPath);
    StringBuilder kept = new StringBuilder();
    for (String line : existing.split("\n", -1)) {
      if (line.startsWith("spannear.")) {
        continue;
      }
      kept.append(line).append('\n');
    }
    String base = kept.toString();
    while (base.endsWith("\n\n")) {
      base = base.substring(0, base.length() - 1);
    }
    Files.writeString(manifestPath, base + out);

    System.out.println("appended spannear.* ground truth to " + manifestPath);
  }

  static void record(
      StringBuilder out,
      String name,
      IndexSearcher searcher,
      String[] terms,
      int slop,
      boolean inOrder)
      throws IOException {
    SpanQuery[] clauses = new SpanQuery[terms.length];
    for (int i = 0; i < terms.length; i++) {
      clauses[i] = new SpanTermQuery(new Term("pos", terms[i]));
    }
    SpanNearQuery query = new SpanNearQuery(clauses, slop, inOrder);
    TopDocs top = searcher.search(query, 100);
    List<Integer> docs = new ArrayList<>();
    for (ScoreDoc sd : top.scoreDocs) {
      docs.add(sd.doc);
    }
    docs.sort(Integer::compare);
    StringBuilder rendered = new StringBuilder();
    for (int doc : docs) {
      if (rendered.length() > 0) rendered.append(',');
      rendered.append(doc);
    }

    String prefix = "spannear." + name + ".";
    out.append(prefix).append("field=pos\n");
    out.append(prefix).append("terms=").append(String.join(" ", terms)).append('\n');
    out.append(prefix).append("slop=").append(slop).append('\n');
    out.append(prefix).append("in_order=").append(inOrder).append('\n');
    out.append(prefix).append("docs=").append(rendered).append('\n');
    // Unused by the port, but it makes the record self-describing.
    out.append(prefix).append("hit_count=").append(docs.size()).append('\n');
    if (!Arrays.asList(terms).isEmpty()) {
      out.append(prefix).append("clauses=").append(terms.length).append('\n');
    }
  }
}
