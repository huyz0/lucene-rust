import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Cross-engine ground truth for `WildcardQuery` (task #34), appended to the
 * already-checked-in {@code fixtures/data/blocktree_index/} directory's {@code
 * manifest.properties} <b>without regenerating the index itself</b> -- same
 * append-only approach {@link AppendDismaxManifest} already uses (opens the
 * existing directory read-only via a real {@link DirectoryReader}/{@link
 * IndexSearcher} rather than re-running {@code GenBlockTree}, which would
 * perturb the segment ID this fixture's committed bytes -- and
 * {@code crates/lucene-ffi/src/query.rs}'s hardcoded {@code segment_id_bytes()}
 * -- depend on).
 *
 * <p>Uses `body`'s already-fixture-known real terms/postings (`cat`={0,2},
 * `dog`={0,1}, `bird`={1,4} -- see {@code GenBlockTree.java}'s own doc
 * comment) against a handful of real {@link WildcardQuery} patterns covering
 * this port's documented glob semantics: a pure literal, a trailing `*`, a
 * `?` single-char wildcard, a bare `*` (matches every term), a pattern that
 * matches nothing, and a `\`-escaped literal `*`/`?` (escaping a
 * non-special byte is also covered, since `body`'s terms contain no literal
 * `*`/`?` byte to escape meaningfully -- the escape mechanism itself is what's
 * under test, not a term that needs it).
 *
 * <p>Idempotent: re-running this tool replaces any previously-appended
 * `wildcard.*` lines rather than duplicating them.
 */
public class AppendWildcardManifest {
  private record Case(String name, String field, String pattern) {}

  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("blocktree_index");
    Path manifestPath = indexDir.resolve("manifest.properties");

    Case[] cases = {
      new Case("literal", "body", "cat"),
      new Case("prefixStar", "body", "b*"),
      new Case("question", "body", "ca?"),
      new Case("bareStar", "body", "*"),
      new Case("noMatch", "body", "zzz*"),
      new Case("escapedStar", "body", "do\\*"),
      new Case("escapedNonSpecial", "body", "do\\g"),
      new Case("questionOnBird", "body", "bir?"),
    };

    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);

      String existing = Files.readString(manifestPath);
      StringBuilder kept = new StringBuilder();
      for (String line : existing.split("\n", -1)) {
        if (line.startsWith("wildcard.")) {
          continue;
        }
        kept.append(line).append('\n');
      }
      String base = kept.toString();
      while (base.endsWith("\n\n")) {
        base = base.substring(0, base.length() - 1);
      }

      StringBuilder out = new StringBuilder(base);
      out.append("wildcard.cases=");
      for (int i = 0; i < cases.length; i++) {
        if (i > 0) {
          out.append(',');
        }
        out.append(cases[i].name());
      }
      out.append('\n');

      for (Case c : cases) {
        WildcardQuery wq = new WildcardQuery(new Term(c.field(), c.pattern()));
        TopDocs td = searcher.search(wq, 10);
        int[] docs =
            Arrays.stream(td.scoreDocs).mapToInt(sd -> sd.doc).sorted().toArray();
        StringBuilder docList = new StringBuilder();
        for (int i = 0; i < docs.length; i++) {
          if (i > 0) {
            docList.append(',');
          }
          docList.append(docs[i]);
        }
        out.append("wildcard.").append(c.name()).append(".field=").append(c.field()).append('\n');
        out.append("wildcard.").append(c.name()).append(".pattern=").append(c.pattern()).append('\n');
        out.append("wildcard.").append(c.name()).append(".docs=").append(docList).append('\n');
      }

      Files.writeString(manifestPath, out.toString());
    }

    System.out.println("appended wildcard.* ground truth to " + manifestPath);
  }
}
