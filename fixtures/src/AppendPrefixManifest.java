import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Cross-engine ground truth for `PrefixQuery` (task #35), appended to the
 * already-checked-in {@code fixtures/data/blocktree_index/} directory's {@code
 * manifest.properties} <b>without regenerating the index itself</b> -- same
 * append-only approach {@link AppendWildcardManifest} already uses (opens the
 * existing directory read-only via a real {@link DirectoryReader}/{@link
 * IndexSearcher} rather than re-running {@code GenBlockTree}, which would
 * perturb the segment ID this fixture's committed bytes -- and
 * {@code crates/lucene-ffi/src/query.rs}'s hardcoded {@code segment_id_bytes()}
 * -- depend on).
 *
 * <p>Uses `body`'s already-fixture-known real terms/postings (`cat`={0,2},
 * `dog`={0,1}, `bird`={1,4} -- see {@code GenBlockTree.java}'s own doc
 * comment) against a handful of real {@link PrefixQuery} cases: a prefix
 * matching one term, a prefix matching multiple terms, an empty prefix
 * (matches every term in the field), a prefix equal to a full existing term
 * (must still match that term, plus any longer terms sharing it), a no-match
 * prefix, and a prefix containing literal `*`/`?` bytes (must be treated as
 * plain literal bytes, never wildcard-interpreted -- `body` has no term
 * actually containing those bytes, so this case is expected to match
 * nothing, but it proves the literal bytes don't get reinterpreted as glob
 * syntax and blow up or match something unrelated).
 *
 * <p>Idempotent: re-running this tool replaces any previously-appended
 * `prefix.*` lines rather than duplicating them.
 */
public class AppendPrefixManifest {
  private record Case(String name, String field, String prefix) {}

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("blocktree_index");
    Path manifestPath = indexDir.resolve("manifest.properties");

    Case[] cases = {
      new Case("singleMatch", "body", "ca"),
      new Case("multiMatch", "body", "b"),
      new Case("empty", "body", ""),
      new Case("fullTerm", "body", "cat"),
      new Case("noMatch", "body", "zzz"),
      new Case("literalWildcardBytes", "body", "a*b?c"),
    };

    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);

      String existing = Files.readString(manifestPath);
      StringBuilder kept = new StringBuilder();
      for (String line : existing.split("\n", -1)) {
        if (line.startsWith("prefix.")) {
          continue;
        }
        kept.append(line).append('\n');
      }
      String base = kept.toString();
      while (base.endsWith("\n\n")) {
        base = base.substring(0, base.length() - 1);
      }

      StringBuilder out = new StringBuilder(base);
      out.append("prefix.cases=");
      for (int i = 0; i < cases.length; i++) {
        if (i > 0) {
          out.append(',');
        }
        out.append(cases[i].name());
      }
      out.append('\n');

      for (Case c : cases) {
        PrefixQuery pq = new PrefixQuery(new Term(c.field(), c.prefix()));
        TopDocs td = searcher.search(pq, 10);
        int[] docs =
            Arrays.stream(td.scoreDocs).mapToInt(sd -> sd.doc).sorted().toArray();
        StringBuilder docList = new StringBuilder();
        for (int i = 0; i < docs.length; i++) {
          if (i > 0) {
            docList.append(',');
          }
          docList.append(docs[i]);
        }
        out.append("prefix.").append(c.name()).append(".field=").append(c.field()).append('\n');
        out.append("prefix.").append(c.name()).append(".prefix=").append(c.prefix()).append('\n');
        out.append("prefix.").append(c.name()).append(".docs=").append(docList).append('\n');
      }

      Files.writeString(manifestPath, out.toString());
    }

    System.out.println("appended prefix.* ground truth to " + manifestPath);
  }
}
