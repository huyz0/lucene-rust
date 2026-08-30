import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.FSDirectory;

/**
 * Generates a single-segment, real-Lucene 10.5.0 index whose {@code t} field holds exactly the term
 * dictionary the automaton/term-intersection benchmarks reason about: {@code t0}..{@code
 * t<n-1>}, one document per term.
 *
 * <p>It exists because that dictionary's *shape* is what the dead-prefix skip in {@code
 * blocktree::FieldTerms::regexp_intersect} is tuned for (see {@code
 * docs/sweep/m2/b8-automata-analysis.md}), and because measuring the skip against an in-memory
 * {@code Vec<&[u8]>} -- which is what b8 could do -- cannot see the win that matters after the
 * {@code c1-lazy-blocktree} batch: not *loading* the `.tim` blocks the skip jumps over. Only a real
 * `Lucene103BlockTreeTermsWriter`-produced dictionary has those blocks; this port's own writer emits
 * one block per leading byte, which for a single-letter-prefixed vocabulary is one giant block.
 *
 * <p>Not wired into {@code scripts/bench-corpus.sh}, which builds the shared query corpus; this is
 * a single-purpose dictionary for the intersect benchmarks. Run it by hand:
 *
 * <pre>{@code
 * CP=fixtures/.jars/lucene-core-10.5.0.jar
 * javac -nowarn -cp "$CP" -d /tmp/gtc benchmarks/corpus/src/GenTermCorpus.java
 * java -cp "$CP:/tmp/gtc" GenTermCorpus benchmarks/.corpus/terms1m 1000000
 * }</pre>
 *
 * <p>Usage: {@code GenTermCorpus <outDir> <numTerms>}
 */
public final class GenTermCorpus {

  public static void main(String[] args) throws IOException {
    if (args.length < 2) {
      System.err.println("usage: GenTermCorpus <outDir> <numTerms>");
      System.exit(2);
    }
    Path out = Paths.get(args[0]);
    int numTerms = Integer.parseInt(args[1]);

    if (Files.exists(out)) {
      deleteRecursively(out);
    }
    Files.createDirectories(out);

    IndexWriterConfig cfg = new IndexWriterConfig(null);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    try (FSDirectory dir = FSDirectory.open(out);
        IndexWriter w = new IndexWriter(dir, cfg)) {
      for (int i = 0; i < numTerms; i++) {
        Document doc = new Document();
        doc.add(new StringField("t", "t" + i, Store.NO));
        w.addDocument(doc);
      }
      w.forceMerge(1);
      w.commit();
    }
    System.out.printf("wrote %d terms to %s%n", numTerms, out);
  }

  private static void deleteRecursively(Path p) throws IOException {
    try (var walk = Files.walk(p)) {
      for (Path q : walk.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(q);
      }
    }
  }
}
