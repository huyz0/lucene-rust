import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.tests.search.DummyTotalHitCountCollector;

/** Same xorshift input, two counters and callback batch as dummy_hit_bench.rs. */
public class DummyTotalHitCountMicro {
  static volatile int sink;
  static final int[] DOCS = new int[4096];
  static {
    int state = 0x12345678;
    for (int i = 0; i < DOCS.length; i++) {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      DOCS[i] = state & 0x7fffffff;
    }
  }
  static long run(long millis) throws Exception {
    long deadline = System.nanoTime() + millis * 1000000;
    long batches = 0;
    do {
      var first = new DummyTotalHitCountCollector();
      var second = new DummyTotalHitCountCollector();
      LeafCollector[] leaves = {first.getLeafCollector(null), second.getLeafCollector(null)};
      for (int doc : DOCS) leaves[doc & 1].collect(doc);
      if (first.getTotalHits() + second.getTotalHits() != 4096) throw new AssertionError("count");
      sink = first.getTotalHits() * 31 + second.getTotalHits();
      batches++;
    } while (System.nanoTime() < deadline);
    return batches * DOCS.length;
  }
  public static void main(String[] args) throws Exception {
    run(Long.parseLong(args[0]));
    long start = System.nanoTime();
    long operations = run(Long.parseLong(args[1]));
    long elapsed = System.nanoTime() - start;
    System.out.printf(java.util.Locale.ROOT, "dummy_hit_callbacks\t%.3f\t%d\t%d%n", (double) elapsed / operations, operations, sink);
  }
}
