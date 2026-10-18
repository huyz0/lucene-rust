import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleDocValuesField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.facet.FacetsCollector;
import org.apache.lucene.facet.FacetsCollectorManager;
import org.apache.lucene.facet.LongValueFacetCounts;
import org.apache.lucene.facet.StringDocValuesReaderState;
import org.apache.lucene.facet.StringValueFacetCounts;
import org.apache.lucene.facet.range.DoubleRange;
import org.apache.lucene.facet.range.DoubleRangeFacetCounts;
import org.apache.lucene.facet.range.LongRange;
import org.apache.lucene.facet.range.LongRangeFacetCounts;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * M13 T13.4d1's value- and range-facet pair (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_value_facets.rs}), with the same case names. {@code build <dir>}
 * writes, once, a 200,000-document, four-segment index: {@code l} a single-valued long (0..9,999),
 * {@code lm} 1-3 longs (0..999), {@code s} one of 2,000 strings ({@code SORTED}) and {@code d} a
 * double (-1..99). Both engines then read those bytes. Every case counts the match-all hits,
 * collected beforehand, and reads the result: {@code value_facet_long} and {@code
 * value_facet_long_multi}, {@code LongValueFacetCounts} over {@code l} and {@code lm} and {@code
 * getTopChildren(10, ..)}; {@code value_facet_string}, {@code StringValueFacetCounts} over {@code
 * s} (its state built beforehand) and {@code getTopChildren(10, "s")}; {@code
 * value_facet_long_range}, {@code LongRangeFacetCounts} of {@code l} into five exclusive ranges;
 * {@code value_facet_long_range_overlap}, of {@code lm} into five overlapping ones; {@code
 * value_facet_double_range}, {@code DoubleRangeFacetCounts} of {@code d} into four ranges (each
 * then {@code getAllChildren}). Units: hits. Each prints a {@code #check} digest of its result
 * first.
 */
public final class ValueFacetsMicro {
  static final int DOCS = 200_000;

  static void build(Path dir) throws IOException {
    Random r = new Random(0x13_4d_2026_1018L);
    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setRAMBufferSizeMB(512);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    try (Directory d = FSDirectory.open(dir.resolve("index"));
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        Document doc = new Document();
        doc.add(new StringField("q", i % 100 == 0 ? "rare" : "common", Field.Store.NO));
        doc.add(new NumericDocValuesField("l", r.nextInt(10_000)));
        int n = 1 + r.nextInt(3);
        for (int k = 0; k < n; k++) doc.add(new SortedNumericDocValuesField("lm", r.nextInt(1000)));
        doc.add(new SortedDocValuesField("s", new BytesRef("v" + r.nextInt(2000))));
        doc.add(new DoubleDocValuesField("d", r.nextDouble() * 100 - 1));
        w.addDocument(doc);
        if ((i + 1) % (DOCS / 4) == 0) w.commit();
      }
    }
    // Last: its presence is what marks the corpus built.
    Files.writeString(dir.resolve("built"), "ok\n");
  }

  static void run(Path dir) throws IOException {
    try (Directory d = FSDirectory.open(dir.resolve("index"));
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      FacetsCollector all = searcher.search(new MatchAllDocsQuery(), new FacetsCollectorManager());
      long hits = FacetCountsMicro.hits(all);
      StringDocValuesReaderState state = new StringDocValuesReaderState(reader, "s");
      LongRange[] exclusive = {
        new LongRange("lt100", Long.MIN_VALUE, true, 100, false),
        new LongRange("100-999", 100, true, 1000, false),
        new LongRange("1k-5k", 1000, true, 5000, false),
        new LongRange("5k-9k", 5000, true, 9000, false),
        new LongRange("ge9k", 9000, true, Long.MAX_VALUE, true),
      };
      LongRange[] overlapping = {
        new LongRange("lt10", 0, true, 10, false),
        new LongRange("lt100", 0, true, 100, false),
        new LongRange("lt500", 0, true, 500, false),
        new LongRange("mid", 250, true, 750, true),
        new LongRange("top", 900, true, 1000, false),
      };
      DoubleRange[] doubles = {
        new DoubleRange("neg", Double.NEGATIVE_INFINITY, true, 0, false),
        new DoubleRange("0-10", 0, true, 10, false),
        new DoubleRange("10-50", 10, true, 50, false),
        new DoubleRange("ge50", 50, true, Double.POSITIVE_INFINITY, true),
      };

      FacetCountsMicro.cases(
          "value_facet_long",
          f -> {
            f.result(new LongValueFacetCounts("l", all).getTopChildren(10, "l"));
            return hits;
          });
      FacetCountsMicro.cases(
          "value_facet_long_multi",
          f -> {
            f.result(new LongValueFacetCounts("lm", all).getTopChildren(10, "lm"));
            return hits;
          });
      FacetCountsMicro.cases(
          "value_facet_string",
          f -> {
            f.result(new StringValueFacetCounts(state, all).getTopChildren(10, "s"));
            return hits;
          });
      FacetCountsMicro.cases(
          "value_facet_long_range",
          f -> {
            f.result(new LongRangeFacetCounts("l", all, exclusive).getAllChildren("l"));
            return hits;
          });
      FacetCountsMicro.cases(
          "value_facet_long_range_overlap",
          f -> {
            f.result(new LongRangeFacetCounts("lm", all, overlapping).getAllChildren("lm"));
            return hits;
          });
      FacetCountsMicro.cases(
          "value_facet_double_range",
          f -> {
            f.result(new DoubleRangeFacetCounts("d", all, doubles).getAllChildren("d"));
            return hits;
          });
    }
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[1]);
    if (args[0].equals("build")) {
      if (Files.exists(dir.resolve("built"))) return;
      Files.createDirectories(dir);
      build(dir);
    } else {
      run(dir);
    }
  }
}
