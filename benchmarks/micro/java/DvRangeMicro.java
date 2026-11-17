import java.io.IOException;
import java.nio.file.Path;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TotalHitCountCollectorManager;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * Doc-values range queries counted over the value-facets corpus (build it with {@code
 * ValueFacetsMicro build <dir>}); the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_dv_range.rs}, with the same case names. Cases: {@code
 * dv_range_long_narrow/mid/wide}: {@code NumericDocValuesField.newSlowRangeQuery} over {@code l}
 * (0..9,999) matching about 1%, 40% and 98%; {@code dv_range_multi_narrow/wide}: {@code
 * SortedNumericDocValuesField.newSlowRangeQuery} over {@code lm} (1-3 values, 0..999) matching
 * about 3% and 80%; {@code dv_range_ord_narrow/wide}: {@code
 * SortedDocValuesField.newSlowRangeQuery} over {@code s} (2,000 strings) matching about 0.5% and
 * 50%. Units: hits.
 */
public final class DvRangeMicro {
  static void count(IndexSearcher s, String name, Query q) throws IOException {
    FacetCountsMicro.cases(name, f -> {
      long n = s.search(q, new TotalHitCountCollectorManager(s.getSlices()));
      f.add(n);
      return n;
    });
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[1]);
    ValueFacetsMicro.main(new String[] {"build", args[1]});
    if (args[0].equals("build")) return;
    try (Directory d = FSDirectory.open(dir.resolve("index"));
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      count(s, "dv_range_long_narrow", NumericDocValuesField.newSlowRangeQuery("l", 100, 199));
      count(s, "dv_range_long_mid", NumericDocValuesField.newSlowRangeQuery("l", 1000, 4999));
      count(s, "dv_range_long_wide", NumericDocValuesField.newSlowRangeQuery("l", 100, 9899));
      count(s, "dv_range_multi_narrow", SortedNumericDocValuesField.newSlowRangeQuery("lm", 10, 19));
      count(s, "dv_range_multi_wide", SortedNumericDocValuesField.newSlowRangeQuery("lm", 100, 899));
      count(
          s,
          "dv_range_ord_narrow",
          SortedDocValuesField.newSlowRangeQuery(
              "s", new BytesRef("v100"), new BytesRef("v104"), true, true));
      count(
          s,
          "dv_range_ord_wide",
          SortedDocValuesField.newSlowRangeQuery(
              "s", new BytesRef("v1"), new BytesRef("v1zzz"), true, true));
    }
  }
}
