import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.FuzzyQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Cross-engine ground truth for {@code FuzzyQuery} across <b>two segments</b>,
 * appended to the already-checked-in {@code multi_segment_scoring_index}
 * without regenerating it (the technique {@link AppendSearchAfterManifest}
 * uses on the same fixture).
 *
 * <p>What it is for. {@code FuzzyQuery}'s default rewrite is
 * {@code MultiTermQuery.TopTermsBlendedFreqScoringRewrite}, and every part of
 * it is <b>reader-wide</b>:
 *
 * <ul>
 *   <li>{@code TermCollectingRewrite.collectTerms} drives <i>one</i>
 *       {@code TopTermsRewrite} collector across every leaf, so the
 *       {@code maxExpansions} queue selects one term set for the whole reader
 *       and a term's {@code docFreq} accumulates across leaves.
 *   <li>{@code BlendedTermQuery.rewrite} then folds
 *       {@code df = Math.max(df, ctx.docFreq())} over those reader-wide
 *       frequencies and scores every clause with it, against the reader-wide
 *       {@code CollectionStatistics.docCount}.
 * </ul>
 *
 * <p>A port that expands per segment picks a different term set in each leaf
 * and blends a different frequency into it. It is invisible on one segment,
 * which is why every other fuzzy fixture in this tree (all over the
 * single-segment {@code blocktree_index}) agrees with a per-segment
 * implementation -- the same blind spot {@link GenMultiSegmentScoring}'s class
 * doc records for {@code TermQuery}'s idf.
 *
 * <p>This fixture's two segments are lopsided on exactly the axis that
 * matters: {@code fox} is in 1 of 4 documents in segment 0 and 3 of 4 in
 * segment 1, {@code dog} in 3 of 4 and 4 of 4, so {@code max(df)} differs
 * per leaf and from the reader-wide value.
 *
 * <p>Per case it records the rewritten query's own selected terms and boosts
 * (walked out of the {@code BooleanQuery} {@code BlendedTermQuery.BOOLEAN_REWRITE}
 * produces), each selected term's reader-wide {@code docFreq}, and real
 * {@code IndexSearcher} {@code TopDocs} as raw float bits in global doc-id
 * space.
 *
 * <p>Idempotent: re-running replaces any previously-appended
 * {@code fuzzymulti.*} lines rather than duplicating them.
 */
public class AppendMultiSegmentFuzzyManifest {

  private record Case(
      String name,
      String field,
      String term,
      int maxEdits,
      int prefixLength,
      boolean transpositions,
      int maxExpansions) {}

  private static final Case[] CASES = {
    // Two selected terms whose reader-wide docFreqs differ, so `max(df)` is a
    // real choice rather than the only number available: `fog` is one edit
    // from both `fox` and `dog`.
    new Case("fog_e1", "body", "fog", 1, 0, true, 50),
    // The same expansion with a wider budget, which also admits `bird`-shaped
    // near misses if any exist.
    new Case("fog_e2", "body", "fog", 2, 0, true, 50),
    // One term only: this isolates the reader-wide `docFreq`/`docCount` from
    // the blending, since with a single term `max(df)` is just that term's.
    new Case("dot_e1", "body", "dot", 1, 0, true, 50),
    new Case("cot_e1", "body", "cot", 1, 0, true, 50),
    // A wide expansion at distance 2 over a four-term vocabulary.
    new Case("bid_e2", "body", "bid", 2, 0, true, 50),
    new Case("dag_e2", "body", "dag", 2, 0, true, 50),
    // `maxExpansions` smaller than the number of matching terms, which is what
    // makes the *selection* reader-wide rather than only the statistics: the
    // queue fills in leaf 0 and the boost it publishes is what leaf 1's
    // `FuzzyTermsEnum` starts from.
    new Case("bid_e2_top1", "body", "bid", 2, 0, true, 1),
    new Case("bid_e2_top2", "body", "bid", 2, 0, true, 2),
    new Case("dag_e2_top1", "body", "dag", 2, 0, true, 1),
    // An exact term, whose boost is 1.0 and whose clause is therefore a bare
    // `TermQuery` in the rewritten query rather than a `BoostQuery`.
    new Case("fox_e1", "body", "fox", 1, 0, true, 50),
    // A prefix requirement, and a transposition-free comparison.
    new Case("dgo_e1_transpositions", "body", "dgo", 1, 0, true, 50),
    new Case("dgo_e1_no_transpositions", "body", "dgo", 1, 0, false, 50),
    new Case("cat_e1_prefix1", "body", "cat", 1, 1, true, 50),
  };

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("multi_segment_scoring_index");
    Path manifestPath = indexDir.resolve("manifest.properties");

    StringBuilder out = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      out.append("fuzzymulti.cases=");
      for (int i = 0; i < CASES.length; i++) {
        if (i > 0) {
          out.append(',');
        }
        out.append(CASES[i].name());
      }
      out.append('\n');
      for (Case c : CASES) {
        record(out, c, searcher, reader);
      }
    }

    String existing = Files.readString(manifestPath);
    StringBuilder kept = new StringBuilder();
    for (String line : existing.split("\n", -1)) {
      if (line.startsWith("fuzzymulti.")) {
        continue;
      }
      kept.append(line).append('\n');
    }
    String base = kept.toString();
    while (base.endsWith("\n\n")) {
      base = base.substring(0, base.length() - 1);
    }
    Files.writeString(manifestPath, base + out);

    System.out.println("appended fuzzymulti.* ground truth to " + manifestPath);
  }

  private static void record(
      StringBuilder out, Case c, IndexSearcher searcher, DirectoryReader reader)
      throws IOException {
    FuzzyQuery query =
        new FuzzyQuery(
            new Term(c.field(), c.term()),
            c.maxEdits(),
            c.prefixLength(),
            c.maxExpansions(),
            c.transpositions());
    String prefix = "fuzzymulti." + c.name() + ".";
    out.append(prefix).append("field=").append(c.field()).append('\n');
    out.append(prefix).append("term=").append(c.term()).append('\n');
    out.append(prefix).append("max_edits=").append(c.maxEdits()).append('\n');
    out.append(prefix).append("prefix_length=").append(c.prefixLength()).append('\n');
    out.append(prefix).append("transpositions=").append(c.transpositions()).append('\n');
    out.append(prefix).append("max_expansions=").append(c.maxExpansions()).append('\n');

    // `TopTermsBlendedFreqScoringRewrite` -> `BlendedTermQuery` ->
    // `BOOLEAN_REWRITE`, i.e. a BooleanQuery of SHOULD
    // `BoostQuery(TermQuery(t), boost_t)` (a bare `TermQuery` when the boost is
    // exactly 1). Walking it out is what pins the *selection*, which a score
    // comparison alone would let a port get right by luck.
    Query rewritten = searcher.rewrite(query);
    StringBuilder terms = new StringBuilder();
    StringBuilder boostBits = new StringBuilder();
    StringBuilder docFreqs = new StringBuilder();
    long blended = 0;
    // `searcher.rewrite` runs to a fixed point, so a one-clause BooleanQuery
    // collapses to the clause itself: the shape is BooleanQuery, or a single
    // BoostQuery/TermQuery, or MatchNoDocsQuery when nothing matched.
    List<Query> clauses = new ArrayList<>();
    if (rewritten instanceof BooleanQuery bq) {
      for (BooleanClause clause : bq.clauses()) {
        if (clause.occur() != BooleanClause.Occur.SHOULD) {
          throw new AssertionError("unexpected occur " + clause.occur());
        }
        clauses.add(clause.query());
      }
    } else if (rewritten instanceof org.apache.lucene.search.MatchNoDocsQuery) {
      // nothing selected
    } else {
      clauses.add(rewritten);
    }
    for (Query q : clauses) {
      float boost = 1.0f;
      if (q instanceof BoostQuery boosted) {
        boost = boosted.getBoost();
        q = boosted.getQuery();
      }
      Term t = ((TermQuery) q).getTerm();
      int df = reader.docFreq(t);
      blended = Math.max(blended, df);
      if (terms.length() > 0) {
        terms.append(',');
        boostBits.append(',');
        docFreqs.append(',');
      }
      terms.append(t.text());
      boostBits.append(Float.floatToIntBits(boost));
      docFreqs.append(df);
    }
    out.append(prefix).append("selected_terms=").append(terms).append('\n');
    out.append(prefix).append("selected_boost_bits=").append(boostBits).append('\n');
    out.append(prefix).append("selected_doc_freqs=").append(docFreqs).append('\n');
    out.append(prefix).append("blended_doc_freq=").append(blended).append('\n');
    out.append(prefix).append("rewritten=").append(rewritten).append('\n');

    // Scores, exactly as `GenMultiSegmentScoring.record` writes them: global
    // doc ids, raw float bits, so nothing passes through decimal rounding.
    TopDocs td = searcher.search(query, 20);
    StringBuilder bits = new StringBuilder();
    for (var sd : td.scoreDocs) {
      if (bits.length() > 0) {
        bits.append(',');
      }
      bits.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
    }
    out.append(prefix).append("bits=").append(bits).append('\n');
  }
}
