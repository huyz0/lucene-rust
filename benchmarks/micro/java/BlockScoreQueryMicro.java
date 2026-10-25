import java.util.Locale;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.tests.search.BlockScoreQueryWrapper;
import org.apache.lucene.util.RamUsageEstimator;

/** Actual Lucene wrapper, same constant-score inputs as the fixture generator. */
public class BlockScoreQueryMicro {
  static final int N = 256;
  static final int[] TARGETS = {0,1,2,4,7,10,15,Integer.MAX_VALUE};
  static volatile long sink;
  static long feed(long hash, long value) { return (hash ^ value) * 1099511628211L; }
  static long[] batch(Weight weight, LeafReaderContext leaf, boolean bounds) throws Exception {
    Scorer[] prepared = bounds ? new Scorer[N] : null;
    if (bounds) for (int i = 0; i < N; i++) prepared[i] = weight.scorer(leaf);
    long digest = 0xcbf29ce484222325L;
    long start = System.nanoTime();
    for (int step = 0; step < N; step++) {
      if (bounds) {
        Scorer s = prepared[step];
        for (int target : TARGETS) {
          int end = s.advanceShallow(target);
          digest = feed(digest, Integer.toUnsignedLong(end));
          digest = feed(digest, Integer.toUnsignedLong(Float.floatToRawIntBits(s.getMaxScore(end))));
          digest = feed(digest, Integer.toUnsignedLong(Float.floatToRawIntBits(s.getMaxScore(Integer.MAX_VALUE))));
          digest = feed(digest, Integer.toUnsignedLong(s.docID()));
        }
      } else {
        Scorer s = weight.scorer(leaf);
        var iterator = s.iterator();
        digest = feed(digest, iterator.cost());
        while (true) {
          int doc = iterator.nextDoc();
          digest = feed(digest, Integer.toUnsignedLong(doc));
          digest = feed(digest, Integer.toUnsignedLong(Float.floatToRawIntBits(s.score())));
          if (doc == Integer.MAX_VALUE) break;
        }
      }
    }
    long elapsed = System.nanoTime() - start;
    sink = digest;
    return new long[] {digest, elapsed};
  }
  public static void main(String[] args) throws Exception {
    long warmup = Long.parseLong(args[0]), measure = Long.parseLong(args[1]);
    try (var dir = new ByteBuffersDirectory()) {
      try (var writer = new IndexWriter(dir, new IndexWriterConfig(new StandardAnalyzer()))) {
        for (int id = 0; id < 16; id++) {
          var d = new Document(); d.add(new StringField("id", Integer.toString(id), Field.Store.NO)); writer.addDocument(d);
        }
        writer.commit();
      }
      try (var reader = DirectoryReader.open(dir)) {
        var searcher = new IndexSearcher(reader); searcher.setQueryCache(null);
        for (int length : new int[] {1,2,3,7}) {
          Query query = new BlockScoreQueryWrapper(GenBlockScoreQueryWrapper.query(false), length);
          Weight weight = searcher.createWeight(searcher.rewrite(query), ScoreMode.COMPLETE, 1);
          var leaf = reader.leaves().get(0);
          if (length == 1) {
            Scorer memoryScorer = weight.scorer(leaf);
            long shallow = RamUsageEstimator.shallowSizeOf(memoryScorer), arrays = 0;
            for (var field : memoryScorer.getClass().getDeclaredFields()) {
              field.setAccessible(true);
              Object value = field.get(memoryScorer);
              if (value instanceof int[] || value instanceof float[]) arrays += RamUsageEstimator.shallowSizeOf(value);
            }
            System.out.printf("#memory\tblock_score_buffer_shallow_bytes\t%d%n", shallow);
            System.out.printf("#memory\tblock_score_buffer_array_bytes\t%d%n", arrays);
          }
          for (boolean bounds : new boolean[] {false,true}) {
            String name = "block_" + (bounds ? "bounds" : "buffer") + "_" + length;
            long expected = batch(weight, leaf, bounds)[0];
            for (int phase = 0; phase < 2; phase++) {
              long start = System.nanoTime(), ns = 0, operations = 0, budget = phase == 0 ? warmup : measure;
              do {
                long[] result = batch(weight, leaf, bounds);
                if (result[0] != expected) throw new AssertionError("unstable digest");
                ns += result[1]; operations += N;
              } while (System.nanoTime() - start < budget * 1000000);
              if (phase == 1) System.out.printf(Locale.ROOT, "%s\t%.3f\t%d%n", name, (double)ns/operations, operations);
            }
            System.out.printf("#check\t%s\t%s\t%d%n", name, Long.toUnsignedString(expected), N);
          }
        }
      }
    }
  }
}
