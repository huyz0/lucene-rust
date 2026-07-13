import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.FuzzyQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Cross-engine ground truth for `FuzzyQuery` (task #42), appended to the
 * already-checked-in {@code fixtures/data/blocktree_index/} directory's
 * {@code manifest.properties} <b>without regenerating the index itself</b> --
 * same append-only approach {@link AppendWildcardManifest}/{@link
 * AppendPrefixManifest} already use (opens the existing directory read-only
 * via a real {@link DirectoryReader}/{@link IndexSearcher} rather than
 * re-running {@code GenBlockTree}, which would perturb the segment ID this
 * fixture's committed bytes -- and {@code crates/lucene-ffi/src/query.rs}'s
 * hardcoded {@code segment_id_bytes()} -- depend on).
 *
 * <p>Uses `body`'s already-fixture-known real terms/postings (`cat`={0,2},
 * `dog`={0,1}, `bird`={1,4} -- see {@code GenBlockTree.java}'s own doc
 * comment) against a handful of real {@link FuzzyQuery} cases, most
 * importantly the transposition case: {@code FuzzyQuery(new Term("body",
 * "cta"), 1, 0, 1, true)} (edit distance 1 from "cat" via one transposition,
 * with transpositions enabled) vs. {@code transpositions=false} at the same
 * {@code maxEdits=1} (must NOT match "cat", since a plain-Levenshtein swap
 * costs 2 edits) -- the exact subtlety real Lucene's default `transpositions
 * = true` behavior hinges on.
 *
 * <p>Idempotent: re-running this tool replaces any previously-appended
 * `fuzzy.*` lines rather than duplicating them.
 */
public class AppendFuzzyManifest {
  private record Case(
      String name, String field, String term, int maxEdits, int prefixLength, boolean transpositions) {}

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("blocktree_index");
    Path manifestPath = indexDir.resolve("manifest.properties");

    Case[] cases = {
      // Exact match (distance 0): "cat" against itself.
      new Case("exactMatch", "body", "cat", 0, 0, true),
      // Single substitution: "cat" -> "cot" is distance 1 ("cot" isn't an
      // indexed term, but "cat" itself is within distance 1 of "cot").
      new Case("singleSubstitution", "body", "cot", 1, 0, true),
      // Single insertion: "ca" -> "cat" is distance 1.
      new Case("singleInsertion", "body", "ca", 1, 0, true),
      // Single deletion: "cats" -> "cat" is distance 1.
      new Case("singleDeletion", "body", "cats", 1, 0, true),
      // Transposition: "cta" is a transposed "cat" -- distance 1 with
      // transpositions enabled, distance 2 (over maxEdits=1) without.
      new Case("transpositionEnabled", "body", "cta", 1, 0, true),
      new Case("transpositionDisabled", "body", "cta", 1, 0, false),
      // prefix_length excludes an otherwise-in-range candidate: target "dat"
      // (not itself an indexed term) is edit-distance 1 from indexed term
      // "cat", well within maxEdits=1 -- but with prefixLength=1, "cat"
      // must share "dat"'s first byte ("d") to match, which it doesn't
      // ("cat" starts with "c"), so this must NOT match despite the
      // in-budget distance.
      new Case("prefixLengthExcludes", "body", "dat", 1, 1, true),
      // Same target/maxEdits with prefixLength=0: no prefix requirement, so
      // "cat" (distance 1 from "dat") does match.
      new Case("prefixLengthZeroNoRequirement", "body", "dat", 1, 0, true),
      // maxEdits boundary: "kitten"-style multi-edit target against "bird"
      // (distance vs "bird" is large) -- use "birdy" (distance 1 from
      // "bird") at maxEdits=1 (matches) vs maxEdits=0 (does not).
      new Case("maxEditsBoundaryAtLimit", "body", "birdy", 1, 0, true),
      new Case("maxEditsBoundaryOverLimit", "body", "birdy", 0, 0, true),
      // No match at all within budget.
      new Case("noMatch", "body", "zzzzzzz", 2, 0, true),
    };

    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);

      String existing = Files.readString(manifestPath);
      StringBuilder kept = new StringBuilder();
      for (String line : existing.split("\n", -1)) {
        if (line.startsWith("fuzzy.")) {
          continue;
        }
        kept.append(line).append('\n');
      }
      String base = kept.toString();
      while (base.endsWith("\n\n")) {
        base = base.substring(0, base.length() - 1);
      }

      StringBuilder out = new StringBuilder(base);
      out.append("fuzzy.cases=");
      for (int i = 0; i < cases.length; i++) {
        if (i > 0) {
          out.append(',');
        }
        out.append(cases[i].name());
      }
      out.append('\n');

      for (Case c : cases) {
        FuzzyQuery fq =
            new FuzzyQuery(
                new Term(c.field(), c.term()),
                c.maxEdits(),
                c.prefixLength(),
                FuzzyQuery.defaultMaxExpansions,
                c.transpositions());
        TopDocs td = searcher.search(fq, 10);
        int[] docs =
            Arrays.stream(td.scoreDocs).mapToInt(sd -> sd.doc).sorted().toArray();
        StringBuilder docList = new StringBuilder();
        for (int i = 0; i < docs.length; i++) {
          if (i > 0) {
            docList.append(',');
          }
          docList.append(docs[i]);
        }
        out.append("fuzzy.").append(c.name()).append(".field=").append(c.field()).append('\n');
        out.append("fuzzy.").append(c.name()).append(".term=").append(c.term()).append('\n');
        out.append("fuzzy.").append(c.name()).append(".maxEdits=").append(c.maxEdits()).append('\n');
        out.append("fuzzy.")
            .append(c.name())
            .append(".prefixLength=")
            .append(c.prefixLength())
            .append('\n');
        out.append("fuzzy.")
            .append(c.name())
            .append(".transpositions=")
            .append(c.transpositions())
            .append('\n');
        out.append("fuzzy.").append(c.name()).append(".docs=").append(docList).append('\n');
      }

      Files.writeString(manifestPath, out.toString());
    }

    System.out.println("appended fuzzy.* ground truth to " + manifestPath);
  }
}
