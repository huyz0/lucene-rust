import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefBuilder;

/**
 * Java side of the term-dictionary seek/scan microbenchmark. The Rust side is
 * {@code crates/lucene-codecs/benches/blocktree_open.rs}; the two pick the same terms in the same
 * order (every 97th term of the widest field, shuffled by the same xorshift64 and truncated to
 * 2000) so the per-seek numbers are directly comparable.
 *
 * <p>It exists because the {@code c1-lazy-blocktree} batch replaced this port's
 * materialize-everything-at-open term dictionary with Lucene's own lazy {@code SegmentTermsEnum}
 * frame stack: that trades a 200x faster open for a slower first seek, and "slower" only means
 * something against what Lucene itself pays for the same work.
 *
 * <p>Emits TSV {@code case<TAB>ns_per_op<TAB>ops} on stdout, for cases {@code seek_hit},
 * {@code seek_miss} and {@code next_all}.
 *
 * <p>Not wired into {@code scripts/bench-micro.sh}, which pairs each Java micro with a case inside
 * {@code benchmarks/rust-runner}'s {@code micro} binary; this one's Rust counterpart is a criterion
 * bench in the codec crate instead. Run it by hand:
 *
 * <pre>{@code
 * CP=fixtures/.jars/lucene-core-10.5.0.jar
 * javac -nowarn -cp "$CP" -d /tmp/tsm benchmarks/micro/java/TermSeekMicro.java
 * taskset -c 2,3 java -cp "$CP:/tmp/tsm" TermSeekMicro benchmarks/.corpus/merged
 * taskset -c 2,3 cargo bench -p lucene-codecs --bench blocktree_open
 * }</pre>
 */
public final class TermSeekMicro {

  public static void main(String[] args) throws IOException {
    if (args.length < 1) {
      System.err.println("usage: TermSeekMicro <index-dir>");
      System.exit(2);
    }
    long warmupMs = Long.getLong("warmupMs", 1500);
    long measureMs = Long.getLong("measureMs", 3000);

    try (MMapDirectory dir = new MMapDirectory(Paths.get(args[0]));
        DirectoryReader reader = DirectoryReader.open(dir)) {
      LeafReader leaf = reader.leaves().get(0).reader();

      // The widest indexed field, the same one the Rust bench picks.
      String fieldName = null;
      long best = -1;
      for (org.apache.lucene.index.FieldInfo fi : leaf.getFieldInfos()) {
        Terms t = leaf.terms(fi.name);
        if (t != null && t.size() > best) {
          best = t.size();
          fieldName = fi.name;
        }
      }
      System.err.printf("TermSeekMicro: field \"%s\" has %d terms%n", fieldName, best);
      final String field = fieldName;

      List<BytesRef> terms = new ArrayList<>();
      TermsEnum te = leaf.terms(field).iterator();
      long n = 0;
      for (BytesRef t = te.next(); t != null; t = te.next()) {
        if (n % 97 == 0) {
          terms.add(BytesRef.deepCopyOf(t));
        }
        n++;
      }
      shuffle(terms);
      terms = new ArrayList<>(terms.subList(0, Math.min(2000, terms.size())));

      List<BytesRef> misses = new ArrayList<>(terms.size());
      for (BytesRef t : terms) {
        BytesRefBuilder b = new BytesRefBuilder();
        b.append(t);
        b.append((byte) '~');
        misses.add(b.toBytesRef());
      }

      run("seek_hit", leaf, field, terms, warmupMs, measureMs);
      run("seek_miss", leaf, field, misses, warmupMs, measureMs);
      runNext("next_all", leaf, field, warmupMs, measureMs);
    }
  }

  /** The Rust bench's xorshift64 shuffle, byte for byte. */
  private static void shuffle(List<BytesRef> v) {
    long state = 0x9E3779B97F4A7C15L;
    for (int i = v.size() - 1; i >= 1; i--) {
      state ^= state << 13;
      state ^= state >>> 7;
      state ^= state << 17;
      int j = (int) Long.remainderUnsigned(state, i + 1);
      BytesRef tmp = v.get(i);
      v.set(i, v.get(j));
      v.set(j, tmp);
    }
  }

  private static long sink;

  private static void run(
      String name, LeafReader leaf, String field, List<BytesRef> targets, long warmupMs, long ms)
      throws IOException {
    loop(leaf, field, targets, warmupMs);
    long start = System.nanoTime();
    long ops = loop(leaf, field, targets, ms);
    long elapsed = System.nanoTime() - start;
    System.out.printf("%s\t%.3f\t%d%n", name, (double) elapsed / ops, ops);
  }

  private static long loop(LeafReader leaf, String field, List<BytesRef> targets, long budgetMs)
      throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long ops = 0;
    long start = System.nanoTime();
    do {
      // A fresh TermsEnum per batch, so a whole batch measures cold seeks the
      // way the Rust side's pooled-per-field scratch does.
      TermsEnum te = leaf.terms(field).iterator();
      for (BytesRef t : targets) {
        if (te.seekExact(t)) {
          sink += te.docFreq();
        }
      }
      ops += targets.size();
    } while (System.nanoTime() - start < budgetNs);
    if (sink == 0xDEADBEEFL) {
      System.err.print("");
    }
    return ops;
  }

  private static void runNext(String name, LeafReader leaf, String field, long warmupMs, long ms)
      throws IOException {
    nextLoop(leaf, field, warmupMs);
    long start = System.nanoTime();
    long ops = nextLoop(leaf, field, ms);
    long elapsed = System.nanoTime() - start;
    System.out.printf("%s\t%.3f\t%d%n", name, (double) elapsed / ops, ops);
  }

  private static long nextLoop(LeafReader leaf, String field, long budgetMs) throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long ops = 0;
    long start = System.nanoTime();
    do {
      TermsEnum te = leaf.terms(field).iterator();
      for (BytesRef t = te.next(); t != null; t = te.next()) {
        sink += t.length + te.docFreq();
        ops++;
      }
    } while (System.nanoTime() - start < budgetNs);
    if (sink == 0xDEADBEEFL) {
      System.err.print("");
    }
    return ops;
  }
}
