import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.RegexpQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Cross-engine ground truth for `RegexpQuery` (task #43), appended to the
 * already-checked-in {@code fixtures/data/blocktree_index/} directory's
 * {@code manifest.properties} <b>without regenerating the index itself</b> --
 * same append-only approach {@link AppendFuzzyManifest}/{@link
 * AppendWildcardManifest} already use (opens the existing directory
 * read-only via a real {@link DirectoryReader}/{@link IndexSearcher} rather
 * than re-running {@code GenBlockTree}, which would perturb the segment ID
 * this fixture's committed bytes -- and
 * {@code crates/lucene-ffi/src/query.rs}'s hardcoded
 * {@code segment_id_bytes()} -- depend on).
 *
 * <p>Uses `body`'s already-fixture-known real terms/postings (`cat`={0,2},
 * `dog`={0,1}, `bird`={1,4} -- see {@code GenBlockTree.java}'s own doc
 * comment) against a handful of real {@link RegexpQuery} cases exercising
 * this port's supported syntax subset (literal, `.`, `*`/`+`/`?`, `[...]`
 * classes, `(...)` alternation) plus, most importantly, the whole-term-match
 * convention: a bare `ca` regexp must NOT match indexed term `cat` (real
 * `RegexpQuery` always matches a term's entire length, never a substring --
 * exactly the subtlety this fixture exists to pin against real Lucene, not
 * just this port's own self-consistent-but-possibly-wrong assumption).
 *
 * <p>Idempotent: re-running this tool replaces any previously-appended
 * `regexp.*` lines rather than duplicating them.
 */
public class AppendRegexpManifest {
  private record Case(String name, String field, String pattern) {}

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("blocktree_index");
    Path manifestPath = indexDir.resolve("manifest.properties");

    Case[] cases = {
      // Exact literal match.
      new Case("exactLiteral", "body", "cat"),
      // Whole-term-match convention: "ca" must NOT match "cat" as a
      // substring.
      new Case("wholeTermNoSubstringMatch", "body", "ca"),
      // "." wildcard: matches "cat"/"dog" (3 bytes) but not "bird" (4 bytes).
      new Case("dotWildcardThreeBytes", "body", "..."),
      // "*" zero-or-more: "do*g" matches "dog" (zero "o" repeats beyond the
      // first) and would also match "doooog" if indexed.
      new Case("starQuantifier", "body", "do*g"),
      // "+" one-or-more: "bi+rd" matches "bird".
      new Case("plusQuantifier", "body", "bi+rd"),
      // "?" zero-or-one: "cats?" matches "cat" (no indexed "cats" term).
      new Case("questionQuantifier", "body", "cats?"),
      // "[...]" character class: "[cb]at" matches "cat" (no "bat" indexed).
      new Case("characterClass", "body", "[cb]at"),
      // "|" alternation across whole terms.
      new Case("alternation", "body", "cat|dog"),
      // Alternation covering all three real terms.
      new Case("alternationAllThree", "body", "cat|dog|bird"),
      // No match at all.
      new Case("noMatch", "body", "zzzzzzz"),
      // Missing field.
      new Case("missingField", "no_such_field", "cat"),
    };

    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);

      String existing = Files.readString(manifestPath);
      StringBuilder kept = new StringBuilder();
      for (String line : existing.split("\n", -1)) {
        if (line.startsWith("regexp.")) {
          continue;
        }
        kept.append(line).append('\n');
      }
      String base = kept.toString();
      while (base.endsWith("\n\n")) {
        base = base.substring(0, base.length() - 1);
      }

      StringBuilder out = new StringBuilder(base);
      out.append("regexp.cases=");
      for (int i = 0; i < cases.length; i++) {
        if (i > 0) {
          out.append(',');
        }
        out.append(cases[i].name());
      }
      out.append('\n');

      for (Case c : cases) {
        RegexpQuery rq = new RegexpQuery(new Term(c.field(), c.pattern()));
        TopDocs td = searcher.search(rq, 10);
        int[] docs = Arrays.stream(td.scoreDocs).mapToInt(sd -> sd.doc).sorted().toArray();
        StringBuilder docList = new StringBuilder();
        for (int i = 0; i < docs.length; i++) {
          if (i > 0) {
            docList.append(',');
          }
          docList.append(docs[i]);
        }
        out.append("regexp.").append(c.name()).append(".field=").append(c.field()).append('\n');
        out.append("regexp.")
            .append(c.name())
            .append(".pattern=")
            .append(c.pattern())
            .append('\n');
        out.append("regexp.").append(c.name()).append(".docs=").append(docList).append('\n');
      }

      Files.writeString(manifestPath, out.toString());
    }

    System.out.println("appended regexp.* ground truth to " + manifestPath);
  }
}
