import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * Java side of the posting-list iteration microbenchmark -- the operation M1 is actually about.
 * The Rust side is {@code benchmarks/rust-runner/src/micro.rs}'s {@code postings_iter} case;
 * {@code scripts/bench-micro.sh --bench postings_iter} runs both over the same index directory
 * and joins them on term.
 *
 * <p>Drives {@code Lucene104PostingsReader}'s {@code BlockPostingsEnum} through the public
 * {@link TermsEnum#postings} API rather than reaching into the codec, because unlike
 * {@code ForUtil} there is nothing package-private in the way: this is how a real query reads
 * postings, so it is what should be measured.
 *
 * <p>Requests {@link PostingsEnum#NONE}, matching the Rust side's doc-only walk. Asking for
 * freqs would decode a {@code PForUtil} block per doc block on this side only and the comparison
 * would be between two different workloads.
 *
 * <p>Emits TSV {@code term<TAB>ns_per_doc<TAB>docs} on stdout.
 */
public final class PostingsIterMicro {

  public static void main(String[] args) throws IOException {
    if (args.length < 1) {
      System.err.println("usage: PostingsIterMicro <index-dir>");
      System.exit(2);
    }
    long warmupMs = Long.getLong("warmupMs", 1500);
    long measureMs = Long.getLong("measureMs", 2000);
    Path index = Paths.get(args[0]);

    try (MMapDirectory dir = new MMapDirectory(index);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      // Same Zipf-ranked spread as the Rust side: t0 is the most frequent term,
      // t2s is well down the tail, so this spans three orders of magnitude of
      // posting-list length rather than measuring one block-encoding shape.
      for (String term : new String[] {"t0", "t1", "tz", "t2s"}) {
        BytesRef ref = new BytesRef(term);
        long docs = walk(reader, ref, warmupMs);
        docs = walk(reader, ref, measureMs);
        if (docs == 0) {
          System.err.println("micro: term \"" + term + "\" has no postings in this index; skipping");
          continue;
        }
        System.out.printf("%s\t%.3f\t%d%n", term, (double) lastNanos / docs, docs);
      }
    }
  }

  private static long lastNanos;

  /**
   * Walks every segment's posting list for {@code term} repeatedly until {@code budgetMs} elapses,
   * returning the total documents visited. Called twice per term: the first call is warmup whose
   * result is discarded, so C2 has compiled and settled before the measured call starts.
   */
  private static long walk(DirectoryReader reader, BytesRef term, long budgetMs)
      throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long visited = 0;
    long start = System.nanoTime();
    long elapsed;
    do {
      for (LeafReaderContext leaf : reader.leaves()) {
        Terms terms = leaf.reader().terms("body");
        if (terms == null) {
          continue;
        }
        TermsEnum te = terms.iterator();
        if (!te.seekExact(term)) {
          continue;
        }
        PostingsEnum pe = te.postings(null, PostingsEnum.NONE);
        while (pe.nextDoc() != PostingsEnum.NO_MORE_DOCS) {
          visited++;
        }
      }
      elapsed = System.nanoTime() - start;
    } while (elapsed < budgetNs);
    lastNanos = elapsed;
    return visited;
  }
}
