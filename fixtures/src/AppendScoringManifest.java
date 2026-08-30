import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MultiPhraseQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Cross-engine BM25 ground truth for the term / boolean / phrase scoring paths,
 * appended to the already-checked-in {@code fixtures/data/blocktree_index/}
 * directory's {@code manifest.properties} <b>without regenerating the index</b>
 * -- same technique, and same reason, as {@link AppendDismaxManifest} (which see
 * for why re-running {@code GenBlockTree} would perturb the committed segment
 * ID that other test suites hardcode).
 *
 * <p>What this records, and why each entry earns its place:
 *
 * <ul>
 *   <li>{@code scoring.term.*} -- a plain {@link TermQuery} on {@code body:cat}.
 *       The most fundamental scored path there is; if this disagrees, every
 *       other score in the port is wrong too. Recorded as exact {@code float}
 *       decimal so a Rust-side comparison can assert bit-for-bit equality via
 *       {@code f32::to_bits}, not a tolerance.
 *   <li>{@code scoring.boolean.should.*} / {@code scoring.boolean.must.*} --
 *       {@code BooleanWeight}'s additive combination over two clauses, the
 *       thing this port's `try_disjunction_lazy`/`try_conjunction_lazy` and its
 *       eager `clause_scores` path all have to agree with.
 *   <li>{@code scoring.boolean.filter*} -- {@code Occur.FILTER}: a required
 *       clause that contributes nothing to the score. The entries cover a
 *       filter alongside a MUST, a filter-only query (score 0), a lone filter
 *       (which {@code BooleanQuery.rewrite} turns into
 *       {@code BoostQuery(ConstantScoreQuery(q), 0)}), a filter with an
 *       optional clause, the {@code minimumNumberShouldMatch} interaction
 *       (filters do not count toward it), a filter duplicating a MUST clause
 *       (dropped by {@code rewrite}), and a filter inside a nested
 *       {@code BooleanQuery}. Bit-exactness is the point: a filter leaking into
 *       the sum, or merely reordering it, moves the last bit and nothing else.
 *   <li>{@code scoring.phrase.exact.*} -- an exact ({@code slop == 0})
 *       {@link PhraseQuery} on the {@code pos} field, whose per-doc frequency
 *       is {@code ExactPhraseMatcher}'s match count.
 *   <li>{@code scoring.phrase.slop2.*} / {@code scoring.phrase.slop3.*} -- the
 *       same phrase at slop 2 and 3, which is the ground truth that separates
 *       {@code SloppyPhraseMatcher}'s {@code sloppyWeight() == 1/(1+matchLength)}
 *       from a naive "a sloppy match contributes frequency 1". Fixture doc 5
 *       has {@code alpha beta} adjacent ({@code matchLength == 0}, weight 1)
 *       while doc 7 has them two positions apart ({@code matchLength == 2},
 *       weight 1/3), so the two documents' recorded scores differ by exactly
 *       that factor inside {@code tf}.
 * </ul>
 *
 * <p>Idempotent: re-running replaces any previously-appended {@code scoring.*}
 * lines rather than duplicating them.
 */
public class AppendScoringManifest {

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("blocktree_index");
    Path manifestPath = indexDir.resolve("manifest.properties");

    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);

      record(out, "scoring.term.cat", searcher, new TermQuery(new Term("body", "cat")));
      record(out, "scoring.term.bird", searcher, new TermQuery(new Term("body", "bird")));

      BooleanQuery should =
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.SHOULD)
              .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.SHOULD)
              .build();
      record(out, "scoring.boolean.should", searcher, should);

      BooleanQuery must =
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.MUST)
              .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.MUST)
              .build();
      record(out, "scoring.boolean.must", searcher, must);

      // ---- Occur.FILTER ------------------------------------------------
      // A filter clause is a MUST clause whose score is dropped. Every entry
      // below exists to pin one way that "dropped" could go wrong, and all of
      // them are compared as raw float bits, because a filter clause leaking
      // into the sum -- or merely being summed in a different order, since f32
      // addition is not associative -- is invisible to any tolerance-based
      // comparison.
      //
      // `+body:cat #body:dog` against the `scoring.boolean.must` entry above
      // (`+body:cat +body:dog`): same matched set, and the filter's dropped
      // score contribution is exactly the difference between the two.
      record(
          out,
          "scoring.boolean.filter",
          searcher,
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.MUST)
              .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.FILTER)
              .build());

      // Filter-only, two clauses: matches the conjunction, scores 0. Two
      // clauses rather than one so it does not hit BooleanQuery.rewrite's
      // single-clause path, which is recorded separately below.
      record(
          out,
          "scoring.boolean.filteronly",
          searcher,
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.FILTER)
              .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.FILTER)
              .build());

      // A lone filter clause: `BooleanQuery.rewrite` turns this into
      // `new BoostQuery(new ConstantScoreQuery(query), 0)` -- "no scoring
      // clauses, so return a score of 0". It matches, at 0; it is NOT a pure
      // negative query.
      record(
          out,
          "scoring.boolean.filter.single",
          searcher,
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.FILTER)
              .build());

      // Filter + optional, minimumNumberShouldMatch == 0: the filter fixes the
      // matched set, the SHOULD clause only adds score where it happens to
      // match. Doc 2 matches `cat` but not `dog`, so it is kept at score 0 --
      // the case that separates "a filter narrows the set" from "an optional
      // clause narrows the set".
      record(
          out,
          "scoring.boolean.filter.should",
          searcher,
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.FILTER)
              .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.SHOULD)
              .build());

      // The minimumNumberShouldMatch interaction: FILTER clauses do not count
      // toward it (only Occur.SHOULD increments BooleanWeight's
      // shouldMatchCount). With minShouldMatch=1 over {dog, bird}, doc 2 --
      // which matches the `cat` filter but neither optional clause -- drops
      // out, which it would not if the filter counted.
      record(
          out,
          "scoring.boolean.filter.minshouldmatch",
          searcher,
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.FILTER)
              .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.SHOULD)
              .add(new TermQuery(new Term("body", "bird")), BooleanClause.Occur.SHOULD)
              .setMinimumNumberShouldMatch(1)
              .build());

      // The same query as `scoring.boolean.must`, plus a FILTER clause that
      // duplicates one of the MUST clauses. `BooleanQuery.rewrite`'s
      // `filters.removeAll(clauseSets.get(Occur.MUST))` drops it, so the score
      // must be bit-identical to `scoring.boolean.must` -- a filter that
      // double-counted its MUST twin would show up here and nowhere else.
      record(
          out,
          "scoring.boolean.filter.dupmust",
          searcher,
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.MUST)
              .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.MUST)
              .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.FILTER)
              .build());

      // A FILTER clause inside a nested BooleanQuery, with a scoring clause at
      // each level: `+(+body:cat #body:dog) body:dog`. Lucene inlines the
      // required inner query and then converts the FILTER-that-is-also-a-SHOULD
      // into a MUST, landing on `+body:cat +body:dog`; a port that executes the
      // nested form directly has to reach the same sum, in the same order.
      record(
          out,
          "scoring.boolean.filter.nested",
          searcher,
          new BooleanQuery.Builder()
              .add(
                  new BooleanQuery.Builder()
                      .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.MUST)
                      .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.FILTER)
                      .build(),
                  BooleanClause.Occur.MUST)
              .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.SHOULD)
              .build());

      // Duplicate MUST/SHOULD clauses, which `BooleanQuery.rewrite` collapses
      // into one clause carrying the *sum* of the duplicates' boosts
      // (10.5.0's two "Deduplicate ... clauses by summing up their boosts"
      // blocks -- note this is a pure structural transform, with no
      // `Similarity` involved; `Similarity.computeQueryTermWeight` is a later
      // addition on Lucene `main` and does not exist in 10.5.0). These two
      // entries are the ground truth for the claim that the collapse is
      // score-neutral: a port that executes the *un*-rewritten query, summing
      // each duplicate clause separately, must land on exactly these bits.
      record(
          out,
          "scoring.boolean.dupshould",
          searcher,
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.SHOULD)
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.SHOULD)
              .add(new TermQuery(new Term("body", "dog")), BooleanClause.Occur.SHOULD)
              .build());
      record(
          out,
          "scoring.boolean.dupmust",
          searcher,
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.MUST)
              .add(new TermQuery(new Term("body", "cat")), BooleanClause.Occur.MUST)
              .build());

      // ---- filter-only pruning (c37) ------------------------------------
      // A filter-only query scores every hit 0, so under a top-`n` collector
      // the queue fills with zeroes and the block-max bound (also 0) can never
      // beat it -- Lucene prunes anyway, because
      // `TopScoreDocCollector.updateMinCompetitiveScore` publishes
      // `Math.nextUp(0f)` and a block whose max score is 0 is below that. These
      // two entries are what a port's pruned run has to agree with: `big`'s
      // term is in 300 documents, so a top-20 search fills the queue 15 times
      // over, and the answer is the 20 lowest doc ids at score 0. The `dup`
      // form is the same query with the clause repeated, which
      // `BooleanQuery.rewrite` collapses -- recorded so a port that executes
      // the *un*-rewritten two-leg conjunction has ground truth for it.
      record(
          out,
          "scoring.boolean.filteronly.big",
          searcher,
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("big", "everywhere")), BooleanClause.Occur.FILTER)
              .build());
      record(
          out,
          "scoring.boolean.filteronly.bigdup",
          searcher,
          new BooleanQuery.Builder()
              .add(new TermQuery(new Term("big", "everywhere")), BooleanClause.Occur.FILTER)
              .add(new TermQuery(new Term("big", "everywhere")), BooleanClause.Occur.FILTER)
              .build());

      record(out, "scoring.phrase.exact", searcher, phrase(0));
      record(out, "scoring.phrase.slop2", searcher, phrase(2));
      record(out, "scoring.phrase.slop3", searcher, phrase(3));

      // MultiPhraseQuery: every position accepts a set of terms
      // (`UnionFullPostingsEnum`), and `MultiPhraseWeight` sums the idf over
      // every term of every position -- both are easy to get wrong by
      // analogy with PhraseQuery, so all three shapes are recorded.
      record(
          out,
          "scoring.multiphrase.union",
          searcher,
          new MultiPhraseQuery.Builder()
              .add(new Term[] {new Term("pos", "alpha")})
              .add(new Term[] {new Term("pos", "beta"), new Term("pos", "alpha")})
              .build());
      record(
          out,
          "scoring.multiphrase.bothslots",
          searcher,
          new MultiPhraseQuery.Builder()
              .add(new Term[] {new Term("pos", "alpha"), new Term("pos", "delta")})
              .add(new Term[] {new Term("pos", "beta"), new Term("pos", "gamma")})
              .build());
      record(
          out,
          "scoring.multiphrase.single",
          searcher,
          new MultiPhraseQuery.Builder()
              .add(new Term[] {new Term("pos", "alpha"), new Term("pos", "delta")})
              .build());
      // The same term listed twice at one position. `UnionPostingsEnum` builds
      // its `PositionsQueue` by draining every sub's positions and sorting --
      // it does **not** deduplicate -- so this records whether real Lucene
      // double-counts the alignment, which is the only way to know whether a
      // port's merged position list may dedup.
      record(
          out,
          "scoring.multiphrase.dup",
          searcher,
          new MultiPhraseQuery.Builder()
              .add(new Term[] {new Term("pos", "alpha"), new Term("pos", "alpha")})
              .add(new Term[] {new Term("pos", "beta")})
              .build());
      record(
          out,
          "scoring.multiphrase.slop2",
          searcher,
          new MultiPhraseQuery.Builder()
              .add(new Term[] {new Term("pos", "alpha")})
              .add(new Term[] {new Term("pos", "beta")})
              .setSlop(2)
              .build());

      // ---- reordered sloppy phrases (c37) --------------------------------
      // `SloppyPhraseMatcher` shifts each slot's positions back by the slot's
      // own index, so a match is a *window* over the shifted positions and the
      // terms may appear in any order within it. This port was in-order only
      // and silently under-matched every transposition; these entries are the
      // ground truth that settles it, over documents the fixture already has:
      //
      //   doc 8555: alpha@0 beta@1     doc 8557: alpha@0 beta@3
      //   doc 8556: alpha@0 alpha@1    doc 8558: delta@0 gamma@1
      //
      // Queried backwards, `"beta alpha"` shifts doc 8555 to beta=1, alpha=-1:
      // a window of width 2, so it matches at slop 2 and not at slop 1, and
      // weighs 1/(1+2). Doc 8557 shifts to beta=3, alpha=-1: width 4.
      record(out, "scoring.phrase.reordered.slop1", searcher, reversedPhrase(1));
      record(out, "scoring.phrase.reordered.slop2", searcher, reversedPhrase(2));
      record(out, "scoring.phrase.reordered.slop4", searcher, reversedPhrase(4));
      // The pair the SpanNearQuery fixture already uses for the same purpose,
      // on terms nothing else touches.
      record(
          out,
          "scoring.phrase.reordered.gammadelta",
          searcher,
          new PhraseQuery.Builder()
              .add(new Term("pos", "gamma"), 0)
              .add(new Term("pos", "delta"), 1)
              .setSlop(2)
              .build());
      // A repeated term at slop 2: `rptGroups` must keep the two slots off one
      // position, so doc 8555 (a single `alpha`) must NOT match however
      // generous the budget, while doc 8556 (`alpha alpha`) must.
      record(
          out,
          "scoring.phrase.repeat.slop2",
          searcher,
          new PhraseQuery.Builder()
              .add(new Term("pos", "alpha"), 0)
              .add(new Term("pos", "alpha"), 1)
              .setSlop(2)
              .build());
      // The same transposition through `MultiPhraseQuery`, whose slots are term
      // *sets* -- a different repeat-detection path in `SloppyPhraseMatcher`.
      record(
          out,
          "scoring.multiphrase.reordered",
          searcher,
          new MultiPhraseQuery.Builder()
              .add(new Term[] {new Term("pos", "beta")})
              .add(new Term[] {new Term("pos", "alpha")})
              .setSlop(2)
              .build());

      // A phrase whose two slots are the SAME term, so the exact matcher's
      // frequency is a real count (doc 6 is "alpha alpha", one match) rather
      // than always 1.
      record(
          out,
          "scoring.phrase.repeat",
          searcher,
          new PhraseQuery.Builder()
              .add(new Term("pos", "alpha"), 0)
              .add(new Term("pos", "alpha"), 1)
              .build());
    }

    String existing = Files.readString(manifestPath);
    StringBuilder kept = new StringBuilder();
    for (String line : existing.split("\n", -1)) {
      if (line.startsWith("scoring.")) {
        continue;
      }
      kept.append(line).append('\n');
    }
    String base = kept.toString();
    while (base.endsWith("\n\n")) {
      base = base.substring(0, base.length() - 1);
    }
    Files.writeString(manifestPath, base + out);

    System.out.println("appended scoring.* ground truth to " + manifestPath);
  }

  /** {@code pos:"beta alpha"~slop} -- the fixture's `alpha beta` documents, backwards. */
  private static PhraseQuery reversedPhrase(int slop) {
    return new PhraseQuery.Builder()
        .add(new Term("pos", "beta"), 0)
        .add(new Term("pos", "alpha"), 1)
        .setSlop(slop)
        .build();
  }

  private static PhraseQuery phrase(int slop) {
    return new PhraseQuery.Builder()
        .add(new Term("pos", "alpha"), 0)
        .add(new Term("pos", "beta"), 1)
        .setSlop(slop)
        .build();
  }

  /**
   * Appends {@code <key>.docScores=doc:score,...} for {@code query}'s top 20 hits,
   * plus {@code <key>.bits=doc:<raw float bits>,...} so a consumer can assert
   * bit-for-bit float equality without going through decimal parsing.
   */
  private static void record(StringBuilder out, String key, IndexSearcher searcher, Query query)
      throws IOException {
    TopDocs td = searcher.search(query, 20);
    StringBuilder scores = new StringBuilder();
    StringBuilder bits = new StringBuilder();
    for (var sd : td.scoreDocs) {
      if (scores.length() > 0) {
        scores.append(',');
        bits.append(',');
      }
      scores.append(sd.doc).append(':').append(sd.score);
      bits.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
    }
    out.append(key).append(".query=").append(query).append('\n');
    out.append(key).append(".docScores=").append(scores).append('\n');
    out.append(key).append(".bits=").append(bits).append('\n');
  }
}
