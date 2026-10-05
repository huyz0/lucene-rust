import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.PriorityQueue;

/**
 * Java side of the terms-aggregation benchmark pair (M10 stage 3); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_aggs.rs}, with the same case names. Over the benchmark corpus
 * ({@code benchmarks/.corpus/merged}, one segment: {@code keyword} {@code SORTED}, {@code cat}
 * {@code SORTED_SET}, {@code num} a {@code LongPoint}), the shape the OpenSearch plugin serves
 * natively -- {@code size: 0}, a filter, a {@code terms} aggregation on a keyword field with the
 * default {@code shard_size} of 25 -- behind a dense filter. Lucene has no terms aggregation, so this
 * side is OpenSearch's: the collector {@code GlobalOrdinalsStringTermsAggregator} gives a keyword
 * field without sub-aggregations (its {@code DenseGlobalOrds} strategy: each match's ordinal read
 * with {@code advanceExact}/{@code ordValue} -- through the singleton for a {@code SORTED_SET} of
 * one value per document -- and its count incremented, in the one pass over the matches), then
 * {@code buildAggregations}' top {@code shard_size} by count desc, ordinal asc, through Lucene's
 * {@code PriorityQueue}, the kept terms read by {@code lookupOrd}. The index is one segment, so the
 * global ordinals are the segment's (OpenSearch's {@code loadGlobal} returns the segment's own
 * values there). Each case prints a {@code #check} digest of its buckets the report compares
 * first.
 */
public final class AggsMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final int SHARD_SIZE = 25;

  interface Op {
    long run() throws IOException;
  }

  static void measure(String name, Op op) throws IOException {
    loop(op, warmupMs);
    long start = System.nanoTime();
    long units = loop(op, measureMs);
    long elapsed = System.nanoTime() - start;
    System.out.printf("%s\t%.3f\t%d%n", name, (double) elapsed / units, units);
    System.out.flush();
  }

  static long loop(Op op, long budgetMs) throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long units = 0;
    long start = System.nanoTime();
    do {
      units += op.run();
    } while (System.nanoTime() - start < budgetNs);
    if (sink == 0xDEADBEEFL) System.err.print("");
    return units;
  }

  /** The aggregation's per-ordinal counts: {@code DenseGlobalOrds}' collect loop. */
  static final class Counts implements Collector {
    final String field;
    long[] counts = new long[0];
    SortedDocValues lastValues;

    Counts(String field) {
      this.field = field;
    }

    @Override
    public LeafCollector getLeafCollector(LeafReaderContext ctx) throws IOException {
      SortedSetDocValues globalOrds = DocValues.getSortedSet(ctx.reader(), field);
      if (counts.length < globalOrds.getValueCount()) {
        counts = Arrays.copyOf(counts, (int) globalOrds.getValueCount());
      }
      final long[] c = counts;
      SortedDocValues single = DocValues.unwrapSingleton(globalOrds);
      if (single != null) {
        lastValues = single;
        return new LeafCollector() {
          @Override
          public void setScorer(Scorable scorer) {}

          @Override
          public void collect(int doc) throws IOException {
            if (single.advanceExact(doc)) {
              c[single.ordValue()]++;
            }
          }
        };
      }
      return new LeafCollector() {
        @Override
        public void setScorer(Scorable scorer) {}

        @Override
        public void collect(int doc) throws IOException {
          if (globalOrds.advanceExact(doc)) {
            for (int i = 0; i < globalOrds.docValueCount(); i++) {
              c[(int) globalOrds.nextOrd()]++;
            }
          }
        }
      };
    }

    @Override
    public ScoreMode scoreMode() {
      return ScoreMode.COMPLETE_NO_SCORES;
    }
  }

  /** {@code buildAggregations}: the digest of the top {@code shard_size} buckets. */
  static long buckets(IndexSearcher s, String field, long[] counts) throws IOException {
    PriorityQueue<long[]> pq =
        new PriorityQueue<>(SHARD_SIZE) {
          @Override
          protected boolean lessThan(long[] a, long[] b) {
            // Least competitive first: the lower count, then the higher ordinal.
            return a[1] != b[1] ? a[1] < b[1] : a[0] > b[0];
          }
        };
    long total = 0;
    for (int ord = 0; ord < counts.length; ord++) {
      long n = counts[ord];
      if (n == 0) continue;
      total += n;
      pq.insertWithOverflow(new long[] {ord, n});
    }
    long[][] kept = new long[pq.size()][];
    long keptDocs = 0;
    for (int i = 0; i < kept.length; i++) {
      kept[i] = pq.pop();
      keptDocs += kept[i][1];
    }
    Arrays.sort(kept, (a, b) -> Long.compare(a[0], b[0]));
    SortedSetDocValues dict = DocValues.getSortedSet(s.getIndexReader().leaves().get(0).reader(), field);
    long h = total - keptDocs;
    for (long[] b : kept) {
      BytesRef term = dict.lookupOrd(b[0]);
      h = h * 31 + b[1] + term.length;
    }
    return h;
  }

  static long aggregate(IndexSearcher s, Query q, String field) throws IOException {
    Counts counts =
        s.search(
            q,
            new CollectorManager<Counts, Counts>() {
              @Override
              public Counts newCollector() {
                return new Counts(field);
              }

              @Override
              public Counts reduce(Collection<Counts> collectors) {
                return collectors.iterator().next();
              }
            });
    return buckets(s, field, counts.counts);
  }

  static Query filter(Query q) {
    return new BooleanQuery.Builder().add(q, BooleanClause.Occur.FILTER).build();
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      String[] names = {"aggs_terms_dense_term", "aggs_terms_mid_term", "aggs_terms_dense_range"};
      Query[] queries = {
        filter(new TermQuery(new Term("body", "t0"))),
        filter(new TermQuery(new Term("body", "t5"))),
        filter(LongPoint.newRangeQuery("num", 0, 700_000)),
      };
      for (String field : List.of("keyword", "cat")) {
        for (int i = 0; i < names.length; i++) {
          String name = names[i] + "_" + field;
          Query q = queries[i];
          System.out.printf("#check\t%s\t%016x\t1%n", name, aggregate(s, q, field));
          measure(
              name,
              () -> {
                sink += aggregate(s, q, field);
                return 1;
              });
        }
      }
    }
  }
}
