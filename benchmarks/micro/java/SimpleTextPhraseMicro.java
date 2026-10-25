import java.io.IOException;
import java.nio.file.Path;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.FSDirectory;

/** Native SimpleText phrases, over the same committed single-segment index as Rust. */
public final class SimpleTextPhraseMicro {
  static volatile long sink;
  interface Op { long run() throws IOException; }

  static long add(long hash, long value) {
    return (hash ^ value) * 0x100000001b3L;
  }

  static final class Digest implements Collector {
    long hash = 0xcbf29ce484222325L;
    long hits;
    @Override public ScoreMode scoreMode() { return ScoreMode.COMPLETE; }
    @Override public LeafCollector getLeafCollector(org.apache.lucene.index.LeafReaderContext ctx) {
      return new LeafCollector() {
        Scorable scorer;
        @Override public void setScorer(Scorable value) { scorer = value; }
        @Override public void collect(int doc) throws IOException {
          hash = add(hash, doc + ctx.docBase);
          hash = add(hash, Integer.toUnsignedLong(Float.floatToRawIntBits(scorer.score())));
          hits++;
        }
      };
    }
  }

  static long run(IndexSearcher searcher, Query query, boolean top) throws IOException {
    if (!top) {
      var collector = new Digest();
      searcher.search(query, collector);
      return add(collector.hash, collector.hits);
    }
    var docs = searcher.search(query, new TopScoreDocCollectorManager(10, 0));
    long hash = 0xcbf29ce484222325L;
    for (var hit : docs.scoreDocs) {
      hash = add(hash, hit.doc);
      hash = add(hash, Integer.toUnsignedLong(Float.floatToRawIntBits(hit.score)));
    }
    return add(hash, docs.scoreDocs.length);
  }

  static long loop(Op operation, long millis) throws IOException {
    long start = System.nanoTime(), count = 0;
    do {
      sink = operation.run();
      count++;
    } while (System.nanoTime() - start < millis * 1_000_000L);
    return count;
  }

  public static void main(String[] args) throws Exception {
    try (var directory = FSDirectory.open(Path.of(args[0]));
        var reader = DirectoryReader.open(directory)) {
      if (reader.leaves().size() != 1) throw new IllegalArgumentException("one segment required");
      var searcher = new IndexSearcher(reader);
      searcher.setQueryCache(null);
      searcher.setSimilarity(new ClassicSimilarity());
      String[] names = {"exact", "sloppy", "repeat"};
      Query[] queries = {
          new PhraseQuery(0, "body", "alpha", "beta"),
          new PhraseQuery(2, "body", "beta", "alpha"),
          new PhraseQuery(3, "body", "alpha", "alpha")};
      for (int i = 0; i < queries.length; i++) {
        for (boolean top : new boolean[] {false, true}) {
          String name = "cx_st_phrase_" + names[i] + (top ? "_top" : "_all");
          if (System.getProperty("microCase") != null
              && !System.getProperty("microCase").equals(name)) continue;
          Query query = queries[i];
          Op operation = () -> run(searcher, query, top);
          System.out.printf("#check\t%s\t%016x\t1%n", name, operation.run());
          loop(operation, Long.getLong("warmupMs", 1500));
          long start = System.nanoTime();
          long count = loop(operation, Long.getLong("measureMs", 2000));
          System.out.printf("%s\t%.3f\t%d%n", name,
              (double) (System.nanoTime() - start) / count, count);
        }
      }
    }
  }
}
