import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanOrQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.queries.spans.SpanWeight;
import org.apache.lucene.queries.spans.Spans;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Cross-engine ground truth for the <b>span extents</b> a {@code SpanQuery}
 * produces -- the sequence of {@code (startPosition(), endPosition())} pairs
 * real Lucene's {@code Spans} iterator emits, per document, per leaf -- not
 * just the hit set {@link AppendSpanNearManifest} already records.
 *
 * <p>Why the hit set is not enough. {@code NearSpansOrdered} and
 * {@code NearSpansUnordered} are <i>iterators whose sub-span cursors only move
 * forward</i>, so the arrangements they visit are a strict subset of "every
 * combination of one span per clause that fits the slop budget":
 *
 * <ul>
 *   <li>{@code NearSpansOrdered.stretchToOrder} advances each later clause to
 *       the first span at or after the previous clause's end and never
 *       rewinds, so each position of clause 0 yields exactly one arrangement.
 *       Its own class doc: "the formed spans only contains minimum slop
 *       matches." A cartesian product also reports the non-minimal ones.
 *   <li>{@code NearSpansUnordered.endPosition()} returns
 *       {@code spanWindow.maxEndPosition}, a <i>running</i> maximum that is
 *       never recomputed when the span that set it moves on, so a reported
 *       extent can end past every span the arrangement actually holds.
 * </ul>
 *
 * <p>Neither difference moves a hit set, which is exactly why this port
 * carried it for three batches: {@code span_doc_ids} only asks whether the
 * span list is non-empty. It moves the <b>extents</b>, and a nested
 * {@code SpanNear}-of-{@code SpanNear} consumes its inner clause's extents --
 * so the {@code nested_*} cases below <i>are</i> hit-set cases.
 *
 * <p>Appended to the already-checked-in {@code multi_segment_scoring_index}
 * <b>without regenerating it</b>, the same technique {@link
 * AppendSearchAfterManifest} uses on the same fixture. That index is the only
 * committed Java-written one whose position lists are rich enough to separate
 * a walk from a product: {@code GenMultiSegmentScoring.longBody} writes 40
 * tokens drawn from {@code {fox, cat, dog, bird}} per document in segment 1,
 * so a term occurs up to twenty times in one document. {@code
 * blocktree_index}'s {@code pos} field has two occurrences per document, where
 * the walk and the product agree on every query.
 *
 * <p>Each case records a query written in a tiny S-expression the Rust test
 * parses too, so the recorded query and the tested query are the same text
 * rather than two hand-kept-in-sync constructions:
 *
 * <pre>
 *   t(field,term)                     SpanTermQuery
 *   n(slop,inOrder,child,child,...)   SpanNearQuery
 *   o(child,child,...)                SpanOrQuery
 * </pre>
 *
 * <p>Idempotent: re-running replaces any previously-appended
 * {@code spanextent.*} lines rather than duplicating them.
 */
public class AppendSpanExtentManifest {

  /** {@code name -> query source}, in the tiny S-expression above. */
  private static final String[][] CASES = {
    // --- ordered: the minimum-slop rule -------------------------------------
    // `fox` is at every even position of segment-1 doc 0; `dog` at 3, 9, 15...
    // With a wide budget a cartesian product pairs one `fox` with several
    // `dog`s; `stretchToOrder` takes only the nearest following one.
    {"ordered_fox_dog_slop0", "n(0,true,t(body,fox),t(body,dog))"},
    {"ordered_fox_dog_slop2", "n(2,true,t(body,fox),t(body,dog))"},
    {"ordered_fox_dog_slop8", "n(8,true,t(body,fox),t(body,dog))"},
    {"ordered_fox_dog_slop30", "n(30,true,t(body,fox),t(body,dog))"},
    // Three clauses, so the greedy advance compounds.
    {"ordered_dog_cat_bird_slop6", "n(6,true,t(body,dog),t(body,cat),t(body,bird))"},
    {"ordered_cat_cat_slop20", "n(20,true,t(body,cat),t(body,cat))"},

    // --- unordered: the running maxEndPosition -------------------------------
    {"unordered_fox_dog_slop0", "n(0,false,t(body,fox),t(body,dog))"},
    {"unordered_fox_dog_slop4", "n(4,false,t(body,fox),t(body,dog))"},
    {"unordered_fox_dog_slop30", "n(30,false,t(body,fox),t(body,dog))"},
    {"unordered_bird_cat_slop10", "n(10,false,t(body,bird),t(body,cat))"},
    {"unordered_repeat_dog_slop3", "n(3,false,t(body,dog),t(body,dog))"},
    {"unordered_triple_slop12", "n(12,false,t(body,fox),t(body,cat),t(body,bird))"},

    // --- SpanOr, whose spans are a plain union ------------------------------
    {"or_cat_dog", "o(t(body,cat),t(body,dog))"},
    {"ordered_or_then_fox_slop3", "n(3,true,o(t(body,cat),t(body,dog)),t(body,fox))"},

    // --- nested: the outer clause consumes the inner clause's extents -------
    // These are the cases where a wrong extent becomes a wrong *hit set*.
    {"nested_ordered_in_ordered", "n(4,true,n(1,true,t(body,dog),t(body,cat)),t(body,bird))"},
    {"nested_ordered_in_unordered", "n(2,false,n(3,true,t(body,fox),t(body,dog)),t(body,bird))"},
    {"nested_unordered_in_ordered", "n(1,true,n(6,false,t(body,cat),t(body,fox)),t(body,dog))"},
    {"nested_unordered_in_unordered", "n(0,false,n(9,false,t(body,bird),t(body,dog)),t(body,cat))"},
    {"nested_tight_outer", "n(0,true,n(20,true,t(body,fox),t(body,bird)),t(body,cat))"},
    {"nested_three_deep", "n(5,true,n(2,true,n(0,true,t(body,dog),t(body,cat)),t(body,bird)),t(body,fox))"},

    // --- nested cases where the extent difference is a *hit set* difference --
    // Found by modelling both algorithms over this fixture's own position
    // lists and searching for a disagreement, then recording real Lucene's
    // answer here. In all three the cartesian product returns segment-1
    // document 0, which Lucene does not: the inner clause's extra,
    // non-minimum-slop extent is one the outer clause can align with.
    // One case per (inner inOrder, outer inOrder) combination that has one.
    {"nested_hitset_unordered_inner", "n(2,true,n(2,false,t(body,fox),t(body,cat)),t(body,cat))"},
    {"nested_hitset_ordered_inner", "n(2,true,n(2,true,t(body,cat),t(body,fox)),t(body,cat))"},
    {"nested_hitset_unordered_outer", "n(0,false,n(2,true,t(body,cat),t(body,fox)),t(body,bird))"},
  };

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("multi_segment_scoring_index");
    Path manifestPath = indexDir.resolve("manifest.properties");

    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      out.append("spanextent.leaf_count=").append(reader.leaves().size()).append('\n');
      out.append("spanextent.cases=");
      for (int i = 0; i < CASES.length; i++) {
        if (i > 0) {
          out.append(',');
        }
        out.append(CASES[i][0]);
      }
      out.append('\n');
      for (String[] c : CASES) {
        record(out, c[0], c[1], searcher, reader);
      }
    }

    String existing = Files.readString(manifestPath);
    StringBuilder kept = new StringBuilder();
    for (String line : existing.split("\n", -1)) {
      if (line.startsWith("spanextent.")) {
        continue;
      }
      kept.append(line).append('\n');
    }
    String base = kept.toString();
    while (base.endsWith("\n\n")) {
      base = base.substring(0, base.length() - 1);
    }
    Files.writeString(manifestPath, base + out);

    System.out.println("appended spanextent.* ground truth to " + manifestPath);
  }

  private static void record(
      StringBuilder out, String name, String source, IndexSearcher searcher, DirectoryReader reader)
      throws IOException {
    SpanQuery query = new Parser(source).parse();
    SpanWeight weight =
        (SpanWeight) query.createWeight(searcher, ScoreMode.COMPLETE_NO_SCORES, 1.0f);
    String prefix = "spanextent." + name + ".";
    out.append(prefix).append("source=").append(source).append('\n');
    for (LeafReaderContext ctx : reader.leaves()) {
      StringBuilder rendered = new StringBuilder();
      Spans spans = weight.getSpans(ctx, SpanWeight.Postings.POSITIONS);
      if (spans != null) {
        int doc;
        while ((doc = spans.nextDoc()) != Spans.NO_MORE_DOCS) {
          StringBuilder perDoc = new StringBuilder();
          int start;
          while ((start = spans.nextStartPosition()) != Spans.NO_MORE_POSITIONS) {
            if (perDoc.length() > 0) {
              perDoc.append(',');
            }
            perDoc.append(start).append('-').append(spans.endPosition());
          }
          if (perDoc.length() == 0) {
            // `nextDoc()` positioned on a document with no emitted span. Java
            // cannot produce this (the two-phase check runs the walk to its
            // first match), so record it loudly rather than silently.
            throw new AssertionError("case " + name + ": doc " + doc + " matched with no spans");
          }
          if (rendered.length() > 0) {
            rendered.append(';');
          }
          rendered.append(doc).append(':').append(perDoc);
        }
      }
      out.append(prefix).append("leaf").append(ctx.ord).append('=').append(rendered).append('\n');
    }
  }

  /** Recursive-descent reader for the S-expression above; the Rust test has a twin. */
  private static final class Parser {
    private final String src;
    private int at;

    Parser(String src) {
      this.src = src;
    }

    SpanQuery parse() {
      SpanQuery q = expr();
      if (at != src.length()) {
        throw new IllegalArgumentException("trailing input in " + src + " at " + at);
      }
      return q;
    }

    private SpanQuery expr() {
      char kind = src.charAt(at);
      expect('t' == kind || 'n' == kind || 'o' == kind, "expected t/n/o");
      at++;
      expect(src.charAt(at) == '(', "expected (");
      at++;
      SpanQuery result;
      if (kind == 't') {
        String field = word();
        expect(src.charAt(at) == ',', "expected ,");
        at++;
        String term = word();
        result = new SpanTermQuery(new Term(field, term));
      } else if (kind == 'n') {
        int slop = Integer.parseInt(word());
        expect(src.charAt(at) == ',', "expected ,");
        at++;
        boolean inOrder = Boolean.parseBoolean(word());
        List<SpanQuery> clauses = new ArrayList<>();
        while (src.charAt(at) == ',') {
          at++;
          clauses.add(expr());
        }
        result = new SpanNearQuery(clauses.toArray(new SpanQuery[0]), slop, inOrder);
      } else {
        List<SpanQuery> clauses = new ArrayList<>();
        clauses.add(expr());
        while (src.charAt(at) == ',') {
          at++;
          clauses.add(expr());
        }
        result = new SpanOrQuery(clauses.toArray(new SpanQuery[0]));
      }
      expect(src.charAt(at) == ')', "expected )");
      at++;
      return result;
    }

    private String word() {
      int start = at;
      while (at < src.length() && (Character.isLetterOrDigit(src.charAt(at)) || src.charAt(at) == '_')) {
        at++;
      }
      expect(at > start, "expected a word");
      return src.substring(start, at);
    }

    private void expect(boolean ok, String what) {
      if (!ok) {
        throw new IllegalArgumentException(what + " in " + src + " at " + at);
      }
    }
  }
}
