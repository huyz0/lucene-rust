import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Cross-engine ground truth for `DisjunctionMaxQuery` (task #32), appended to the
 * already-checked-in {@code fixtures/data/blocktree_index/} directory's
 * {@code manifest.properties} <b>without regenerating the index itself</b>.
 *
 * <p>Unlike {@link GenBlockTree}, which writes a brand-new segment (and therefore a
 * brand-new random segment ID -- real Lucene's {@code IndexWriter} assigns one via
 * {@code StringHelper.randomId()} on every run, with no seed knob) every time it's
 * run, this tool opens the existing on-disk directory read-only via a real
 * {@link DirectoryReader} and runs a real {@link DisjunctionMaxQuery} against it.
 * Re-running {@code GenBlockTree} to add this ground truth would perturb the
 * segment ID baked into that fixture's already-committed `.si`/`.doc`/etc bytes,
 * which `lucene-ffi`'s test suite hardcodes (see
 * {@code crates/lucene-ffi/src/query.rs}'s {@code segment_id_bytes()}) -- this tool
 * sidesteps that hazard entirely by never touching the index, only appending plain
 * text lines to `manifest.properties`.
 *
 * <p>Uses `body`'s already-fixture-known real postings (`cat`={0,2}, `dog`={0,1},
 * see `GenBlockTree.java`'s own doc comment) and a real
 * {@code tieBreakerMultiplier=0.3f}, recording real Lucene's own `TopDocs` (doc,
 * score) pairs -- the cross-engine proof this port's dismax formula
 * (`max(disjunct scores) + tie_breaker * sum(rest)`) is checked against, not a
 * hand-derived expectation, per the differential-testing skill.
 *
 * <p>Idempotent: re-running this tool replaces any previously-appended `dismax.*`
 * lines rather than duplicating them.
 */
public class AppendDismaxManifest {
  public static void main(String[] args) throws IOException {
    Path indexDir = Path.of(args[0]).resolve("blocktree_index");
    Path manifestPath = indexDir.resolve("manifest.properties");

    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);

      float tieBreaker = 0.3f;
      DisjunctionMaxQuery dmq =
          new DisjunctionMaxQuery(
              List.of(
                  new TermQuery(new Term("body", "cat")),
                  new TermQuery(new Term("body", "dog"))),
              tieBreaker);
      TopDocs td = searcher.search(dmq, 10);

      StringBuilder docScores = new StringBuilder();
      for (var sd : td.scoreDocs) {
        if (docScores.length() > 0) {
          docScores.append(',');
        }
        docScores.append(sd.doc).append(':').append(sd.score);
      }

      String existing = Files.readString(manifestPath);
      StringBuilder kept = new StringBuilder();
      for (String line : existing.split("\n", -1)) {
        if (line.startsWith("dismax.")) {
          continue;
        }
        if (!line.isEmpty() || kept.length() == 0) {
          // Preserve non-dismax lines verbatim (including the trailing empty
          // split segment from the file's final newline, handled below).
        }
        kept.append(line).append('\n');
      }
      // The loop above appends an extra trailing '\n' for the final empty
      // split segment; trim it back to exactly one trailing newline.
      String base = kept.toString();
      while (base.endsWith("\n\n")) {
        base = base.substring(0, base.length() - 1);
      }

      StringBuilder out = new StringBuilder(base);
      out.append("dismax.tieBreaker=").append(tieBreaker).append('\n');
      out.append("dismax.termA.field=body\n");
      out.append("dismax.termA.term=cat\n");
      out.append("dismax.termB.field=body\n");
      out.append("dismax.termB.term=dog\n");
      out.append("dismax.realLuceneDocScores=").append(docScores).append('\n');

      Files.writeString(manifestPath, out.toString());
    }

    System.out.println("appended dismax.* ground truth to " + manifestPath);
  }
}
