import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.tests.search.BulkScorerWrapperScorer;
import org.apache.lucene.util.Bits;

/** Actual Lucene wrapper; setup and teardown are outside traversal timing. */
public class BulkScorerWrapperMicro {
  static volatile long sink;
  static final class Source extends BulkScorer {
    final int[] docs;
    int index;
    Source(int[] docs) { this.docs = docs; }
    public long cost() { return docs.length; }
    public int score(LeafCollector collector, Bits accept, int min, int max) throws java.io.IOException {
      if (accept != null) throw new AssertionError();
      var value = new Scorable() {
        public float score() { return Float.intBitsToFloat(0x3f000000 + (docs[index] & 1023)); }
      };
      collector.setScorer(value);
      while (index < docs.length && docs[index] < min) ++index;
      while (index < docs.length && docs[index] < max) { collector.collect(docs[index]); ++index; }
      return index == docs.length ? DocIdSetIterator.NO_MORE_DOCS : max;
    }
  }
  static long mix(long digest, int doc, int bits) { return (digest * 31 + doc) * 31 + bits; }
  static long[] expected(int[] docs, boolean advance) {
    long digest = 0; int count = 0, target = 0;
    for (int doc : docs) {
      if (!advance || doc >= target) {
        digest = mix(digest, doc, 0x3f000000 + (doc & 1023));
        ++count; target = doc + 17;
      }
    }
    return new long[] {digest, count};
  }
  static double run(int[] docs, boolean advance, long millis) throws Exception {
    long[] expected = expected(docs, advance);
    long elapsed = 0, batches = 0;
    while (elapsed < millis * 1_000_000) {
      var wrapper = new BulkScorerWrapperScorer(new Source(docs), 64);
      var iterator = wrapper.iterator();
      if (iterator.cost() != docs.length) throw new AssertionError();
      long digest = 0; int count = 0, target = 0;
      long start = System.nanoTime();
      for (;;) {
        int doc = advance ? iterator.advance(target) : iterator.nextDoc();
        if (doc == DocIdSetIterator.NO_MORE_DOCS) break;
        digest = mix(digest, doc, Float.floatToRawIntBits(wrapper.score()));
        ++count; target = doc + 17;
      }
      elapsed += System.nanoTime() - start;
      if (digest != expected[0] || count != expected[1]) throw new AssertionError();
      sink = digest; ++batches;
    }
    return (double) elapsed / batches;
  }
  public static void main(String[] args) throws Exception {
    long warmup = args.length > 0 ? Long.parseLong(args[0]) : 500;
    long measure = args.length > 1 ? Long.parseLong(args[1]) : 750;
    for (int gap : new int[] {1, 17}) {
      int[] docs = new int[8192];
      for (int i = 0; i < docs.length; ++i) docs[i] = i * gap;
      for (boolean advance : new boolean[] {false, true}) {
        String name = "bulk_wrapper_" + (gap == 1 ? "dense" : "sparse") + (advance ? "_advance" : "_next");
        run(docs, advance, warmup);
        double ns = run(docs, advance, measure);
        long[] check = expected(docs, advance);
        System.out.printf(java.util.Locale.ROOT, "%s\t%.3f%n", name, ns);
        System.out.println("#check\t" + name + "\t" + Long.toUnsignedString(check[0]) + "\t" + check[1]);
      }
    }
    System.out.println("bulk_wrapper_buffer_payload_bytes\t512");
  }
}
