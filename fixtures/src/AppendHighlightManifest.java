import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.uhighlight.OffsetsEnum;
import org.apache.lucene.search.uhighlight.PhraseHelper;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Cross-engine ground truth for the **highlighter's** phrase enumeration --
 * `UnifiedHighlighter`'s {@link PhraseHelper#createOffsetsEnumsForSpans}, the
 * position-sensitive half of `FieldOffsetStrategy` -- appended to the
 * already-checked-in {@code fixtures/data/blocktree_index/} directory's
 * {@code manifest.properties} <b>without regenerating the index</b> (same
 * technique and same reason as {@link AppendScoringManifest}).
 *
 * <p>Why this exists: batch c37 ported {@code SloppyPhraseMatcher} into the
 * search path, leaving {@code highlighter::phrase_match_offsets} as the only
 * in-order-only phrase path in the port -- so a reordered sloppy match was
 * scored but offered no highlight. The obvious fix ("enumerate through the
 * sloppy matcher") would have been <b>wrong</b>, and only real Lucene says so:
 * the highlighter does not use {@code SloppyPhraseMatcher} at all.
 * {@code WeightedSpanTermExtractor.extract} turns a {@link PhraseQuery} into
 *
 * <pre>{@code
 * boolean inorder = (phraseQuery.getSlop() == 0);
 * new SpanNearQuery(clauses, phraseQuery.getSlop() + positionGaps, inorder)
 * }</pre>
 *
 * <p>and {@code NearSpansUnordered}'s slop is a different quantity from
 * {@code PhraseQuery}'s: its window test is
 * {@code maxEndPosition - top().startPosition() - totalSpanLength <= slop},
 * i.e. {@code max(p) - min(p) + 1 - n} for n one-position term spans, where
 * {@code SloppyPhraseMatcher}'s {@code matchLength} is
 * {@code max(p_i - i) - min(p_i - i)}. For the reordered pair {@code alpha@0
 * beta@1} queried as {@code "beta alpha"} those are <b>0</b> and <b>2</b>. At
 * slop 0 the difference is invisible, because {@code inorder} is forced there
 * and {@code NearSpansOrdered} rejects the transposition outright -- so
 * {@code highlight.reordered_slop0.offsets} is empty. At <b>slop 1</b> it is
 * not: the highlighter marks the span while the scorer does not match the
 * document at all until slop 2. Recording Lucene's own answer is the only way
 * to get that right, which is what these entries are.
 *
 * <p>Each recorded case is {@code highlight.<name>.*}:
 * {@code doc}, {@code field}, {@code phrase} (space-separated terms in phrase
 * order), {@code slop}, and {@code offsets} -- the {@link OffsetsEnum}s
 * {@code createOffsetsEnumsForSpans} produced, rendered as
 * {@code term:start,end;...} in {@code OffsetsEnum.compareTo} order (start,
 * then end, then term), which is the order {@code FieldOffsetStrategy} hands to
 * the passage formatter. An empty value means "no highlight at all".
 *
 * <p>Idempotent: re-running replaces any previously-appended {@code highlight.*}
 * lines rather than duplicating them.
 */
public class AppendHighlightManifest {

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("blocktree_index");
    Path manifestPath = indexDir.resolve("manifest.properties");

    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      LeafReader leaf = reader.leaves().get(0).reader();

      // doc 8555: alpha@0 [0,5)  beta@1 [6,10)
      // doc 8556: alpha@0 [0,5)  alpha@1 [6,11)
      // doc 8557: alpha@0 [0,5)  beta@3 [12,16)
      // doc 8558: delta@0 [20,25) gamma@1 [26,31)
      //
      // In phrase order, so "beta alpha" is a *reordered* query against 8555.

      // The exact case both engines already agree on -- the control.
      record(out, "exact", leaf, 8555, "pos", new String[] {"alpha", "beta"}, 0);
      // A reordered pair. c37 proved the *scorer* needs slop 2 here; these
      // entries are what the *highlighter* does at each slop.
      record(out, "reordered_slop0", leaf, 8555, "pos", new String[] {"beta", "alpha"}, 0);
      record(out, "reordered_slop1", leaf, 8555, "pos", new String[] {"beta", "alpha"}, 1);
      record(out, "reordered_slop2", leaf, 8555, "pos", new String[] {"beta", "alpha"}, 2);
      // The same transposition on terms nothing else touches, and with a
      // wider real gap between the two occurrences.
      record(out, "reordered_gammadelta", leaf, 8558, "pos", new String[] {"gamma", "delta"}, 2);
      record(out, "gap_in_order_slop0", leaf, 8557, "pos", new String[] {"alpha", "beta"}, 0);
      record(out, "gap_in_order_slop2", leaf, 8557, "pos", new String[] {"alpha", "beta"}, 2);
      record(out, "gap_reordered_slop2", leaf, 8557, "pos", new String[] {"beta", "alpha"}, 2);
      record(out, "gap_reordered_slop4", leaf, 8557, "pos", new String[] {"beta", "alpha"}, 4);
      // A repeated term: two slots, one term, one output enum -- and
      // SpanNearQuery has no rptGroups, so whether the two clauses may settle
      // on the SAME position is a real question only Lucene can answer.
      record(out, "repeat_two_occurrences", leaf, 8556, "pos", new String[] {"alpha", "alpha"}, 0);
      record(out, "repeat_single_occurrence", leaf, 8555, "pos", new String[] {"alpha", "alpha"}, 2);
      // A phrase term absent from the document: no spans, no highlight.
      record(out, "absent_term", leaf, 8555, "pos", new String[] {"alpha", "gamma"}, 2);
      // A one-term "phrase": WeightedSpanTermExtractor turns it into a bare
      // SpanTermQuery, so every occurrence is highlighted.
      record(out, "single_term", leaf, 8556, "pos", new String[] {"alpha"}, 0);
    }

    String existing = Files.readString(manifestPath);
    StringBuilder kept = new StringBuilder();
    for (String line : existing.split("\n", -1)) {
      if (line.startsWith("highlight.")) {
        continue;
      }
      kept.append(line).append('\n');
    }
    String base = kept.toString();
    while (base.endsWith("\n\n")) {
      base = base.substring(0, base.length() - 1);
    }
    Files.writeString(manifestPath, base + out);

    System.out.println("appended highlight.* ground truth to " + manifestPath);
  }

  static void record(
      StringBuilder out,
      String name,
      LeafReader leaf,
      int docId,
      String field,
      String[] phrase,
      int slop)
      throws IOException {
    PhraseQuery.Builder b = new PhraseQuery.Builder();
    for (int i = 0; i < phrase.length; i++) {
      b.add(new Term(field, phrase[i]), i);
    }
    b.setSlop(slop);
    Query query = b.build();

    // The PhraseHelper `UnifiedHighlighter` builds for a query it must
    // highlight positionally: `requireFieldMatch` on the one field, no
    // rewriting, and `WeightedSpanTermExtractor`'s default query conversion.
    PhraseHelper helper =
        new PhraseHelper(
            query, field, (f) -> f.equals(field), (spanQuery) -> Boolean.FALSE, (q) -> null, false);

    List<OffsetsEnum> enums = new ArrayList<>();
    helper.createOffsetsEnumsForSpans(leaf, docId, enums);

    // `OffsetsEnum.compareTo` order: start offset, then end offset, then term.
    List<String> spans = new ArrayList<>();
    for (OffsetsEnum e : enums) {
      while (e.nextPosition()) {
        spans.add(
            String.format(
                "%08d:%08d:%s", e.startOffset(), e.endOffset(), e.getTerm().utf8ToString()));
      }
    }
    Collections.sort(spans);
    StringBuilder rendered = new StringBuilder();
    for (String s : spans) {
      String[] parts = s.split(":");
      if (rendered.length() > 0) rendered.append(';');
      rendered
          .append(parts[2])
          .append(':')
          .append(Integer.parseInt(parts[0]))
          .append(',')
          .append(Integer.parseInt(parts[1]));
    }

    String prefix = "highlight." + name + ".";
    out.append(prefix).append("doc=").append(docId).append('\n');
    out.append(prefix).append("field=").append(field).append('\n');
    out.append(prefix).append("phrase=").append(String.join(" ", phrase)).append('\n');
    out.append(prefix).append("slop=").append(slop).append('\n');
    out.append(prefix).append("offsets=").append(rendered).append('\n');
  }
}
