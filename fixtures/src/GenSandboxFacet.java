import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleDocValuesField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.facet.DrillDownQuery;
import org.apache.lucene.facet.DrillSideways;
import org.apache.lucene.facet.FacetField;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.FacetsCollector;
import org.apache.lucene.facet.FacetsCollectorManager;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.LabelAndValue;
import org.apache.lucene.facet.MultiDoubleValuesSource;
import org.apache.lucene.facet.MultiLongValuesSource;
import org.apache.lucene.facet.range.DoubleRange;
import org.apache.lucene.facet.range.LongRange;
import org.apache.lucene.facet.taxonomy.FacetLabel;
import org.apache.lucene.facet.taxonomy.TaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyWriter;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.internal.hppc.LongIntHashMap;
import org.apache.lucene.sandbox.facet.FacetFieldCollector;
import org.apache.lucene.sandbox.facet.FacetFieldCollectorManager;
import org.apache.lucene.sandbox.facet.cutters.LongValueFacetCutter;
import org.apache.lucene.sandbox.facet.cutters.TaxonomyFacetsCutter;
import org.apache.lucene.sandbox.facet.iterators.CandidateSetOrdinalIterator;
import org.apache.lucene.sandbox.facet.iterators.LengthOrdinalIterator;
import org.apache.lucene.sandbox.facet.iterators.OrdinalIterator;
import org.apache.lucene.sandbox.facet.iterators.TaxonomyChildrenOrdinalIterator;
import org.apache.lucene.sandbox.facet.iterators.TopnOrdinalIterator;
import org.apache.lucene.sandbox.facet.labels.RangeOrdToLabel;
import org.apache.lucene.sandbox.facet.labels.TaxonomyOrdLabelBiMap;
import org.apache.lucene.sandbox.facet.plain.histograms.HistogramCollectorManager;
import org.apache.lucene.sandbox.facet.recorders.CountFacetRecorder;
import org.apache.lucene.sandbox.facet.recorders.LongAggregationsFacetRecorder;
import org.apache.lucene.sandbox.facet.recorders.MultiFacetsRecorder;
import org.apache.lucene.sandbox.facet.recorders.Reducer;
import org.apache.lucene.sandbox.facet.utils.CommonFacetBuilder;
import org.apache.lucene.sandbox.facet.utils.ComparableUtils;
import org.apache.lucene.sandbox.facet.utils.DrillSidewaysFacetOrchestrator;
import org.apache.lucene.sandbox.facet.utils.FacetBuilder;
import org.apache.lucene.sandbox.facet.utils.FacetOrchestrator;
import org.apache.lucene.sandbox.facet.utils.LongValueFacetBuilder;
import org.apache.lucene.sandbox.facet.utils.PostCollectionFaceting;
import org.apache.lucene.sandbox.facet.utils.RangeFacetBuilderFactory;
import org.apache.lucene.sandbox.facet.utils.TaxonomyFacetBuilder;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TotalHitCountCollectorManager;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.IOUtils;

/**
 * lucene-sandbox's facet package ({@code org.apache.lucene.sandbox.facet}), recorded from Lucene
 * 10.5.0 for crates/lucene-search/tests/sandbox_facet_fixtures.rs. Written to {@code
 * sandbox_facet/}:
 *
 * <pre>
 *   index/, taxo/    three segments (1500, 1500 and 300 documents; deletions in the first and
 *                    last) and their taxonomy
 *   builders.tsv     query, builder: every FacetOrchestrator builder's FacetResult
 *   sideways.tsv     drill-down, builder: DrillSidewaysFacetOrchestrator's results and hit count
 *   recorders.tsv    query, case: the cutters, recorders, iterators and labels used directly
 *   post.tsv         query, case: PostCollectionFaceting over FacetsCollector results
 *   histograms.tsv   query, field, width, maxBuckets: HistogramCollectorManager's buckets
 *   errors.tsv       case, the exception's class and message
 * </pre>
 */
public class GenSandboxFacet {
  static final int[] SEGMENTS = {1500, 1500, 300};

  static FacetsConfig config() {
    FacetsConfig c = new FacetsConfig();
    c.setMultiValued("Author", false);
    c.setHierarchical("Publish Date", true);
    c.setMultiValued("Tag", true);
    c.setMultiValued("Rating", true);
    c.setRequireDimCount("Rating", true);
    c.setIndexFieldName("Color", "$color");
    c.setHierarchical("Color", true);
    return c;
  }

  static Document doc(int i) {
    Document d = new Document();
    d.add(new StringField("id", "" + i, Field.Store.NO));
    d.add(new StringField("q", i % 2 == 0 ? "even" : "odd", Field.Store.NO));
    if (i % 97 == 0) d.add(new StringField("q", "rare", Field.Store.NO));
    if ((i < 1500 && i % 11 == 0) || (i >= 3000 && i % 7 == 0)) {
      d.add(new StringField("del", "y", Field.Store.NO));
    }
    // Taxonomy dims.
    if (i % 31 != 0) d.add(new FacetField("Author", "a" + (i * 7 % 17)));
    d.add(new FacetField("Publish Date", "" + (2000 + i % 4), "" + (1 + i % 12)));
    d.add(new FacetField("Tag", "t" + (i % 5)));
    if ((i * 3) % 8 != i % 5) d.add(new FacetField("Tag", "t" + ((i * 3) % 8)));
    d.add(new FacetField("Rating", "r" + (i % 3)));
    d.add(new FacetField("Color", "c" + (i % 3), "s" + (i % 2)));
    // Multi-valued longs: duplicates (consecutive after sorting), negatives.
    if (i % 13 != 0) {
      d.add(new SortedNumericDocValuesField("price", i % 50));
      d.add(new SortedNumericDocValuesField("price", (i * 7) % 90 - 20));
      if (i % 10 == 0) d.add(new SortedNumericDocValuesField("price", i % 50));
    }
    // Single-valued long with points (the histogram's bulk path).
    d.add(new LongPoint("pt", i % 1000));
    d.add(new NumericDocValuesField("pt", i % 1000));
    // Single-valued int with points.
    d.add(new IntPoint("ipt", (i * 17) % 600 - 300));
    d.add(new NumericDocValuesField("ipt", (i * 17) % 600 - 300));
    // Single-valued with a skip index, no points.
    if (i % 9 != 0) d.add(NumericDocValuesField.indexedField("skp", (i * 13) % 5000));
    // A single-valued long for range facets and aggregations.
    d.add(new NumericDocValuesField("single", (i * 37) % 1000 - 100));
    // Doubles: multi-valued sortable longs, and a single raw-bits field.
    d.add(
        new SortedNumericDocValuesField(
            "weight", org.apache.lucene.util.NumericUtils.doubleToSortableLong((i % 40) * 0.25 - 2)));
    if (i % 4 == 0) {
      d.add(
          new SortedNumericDocValuesField(
              "weight", org.apache.lucene.util.NumericUtils.doubleToSortableLong(i % 7 + 0.5)));
    }
    d.add(new DoubleDocValuesField("dweight", (i % 23) * 1.5 - 10));
    d.add(new SortedDocValuesField("kw", new org.apache.lucene.util.BytesRef("k" + (i % 4))));
    return d;
  }

  static String esc(String s) {
    StringBuilder out = new StringBuilder();
    for (char c : s.toCharArray()) {
      if (c < 0x20 || c == '\\') out.append(String.format("\\u%04x", (int) c));
      else out.append(c);
    }
    return out.toString();
  }

  static String render(FacetResult r) {
    StringBuilder sb = new StringBuilder();
    sb.append(esc(r.dim)).append('|').append(esc(String.join("/", r.path))).append('|');
    sb.append(r.value).append('|').append(r.childCount).append('|');
    for (int i = 0; i < r.labelValues.length; i++) {
      LabelAndValue lv = r.labelValues[i];
      if (i > 0) sb.append(',');
      sb.append(esc(lv.label)).append('=').append(lv.value).append('/').append(lv.count);
    }
    return sb.toString();
  }

  static String err(Throwable t) {
    return t.getClass().getSimpleName() + ": " + t.getMessage();
  }

  static Map<String, Query> bases() {
    Map<String, Query> m = new LinkedHashMap<>();
    m.put("all", new MatchAllDocsQuery());
    m.put("even", new TermQuery(new Term("q", "even")));
    m.put("rare", new TermQuery(new Term("q", "rare")));
    m.put("none", new TermQuery(new Term("q", "absent")));
    return m;
  }

  static LongRange[] disjointRanges() {
    return new LongRange[] {
      new LongRange("neg", -20, true, -1, true),
      new LongRange("low", 0, true, 10, false),
      new LongRange("mid", 10, true, 30, true),
      new LongRange("top", 40, false, Long.MAX_VALUE, true),
    };
  }

  static LongRange[] overlappingRanges() {
    return new LongRange[] {
      new LongRange("all", Long.MIN_VALUE, true, Long.MAX_VALUE, true),
      new LongRange("a", 0, true, 20, true),
      new LongRange("b", 10, true, 40, true),
      new LongRange("c", 20, true, 20, true),
      new LongRange("d", -5, true, 5, false),
      new LongRange("e", 500, true, 900, true),
      new LongRange("f", 15, true, 15, true),
    };
  }

  static DoubleRange[] doubleRanges() {
    return new DoubleRange[] {
      new DoubleRange("neg", Double.NEGATIVE_INFINITY, true, 0, false),
      new DoubleRange("zero_two", 0, true, 2, true),
      new DoubleRange("one_five", 1, false, 5, true),
      new DoubleRange("big", 5, false, Double.POSITIVE_INFINITY, true),
    };
  }

  static List<Map.Entry<String, FacetBuilder>> builders(
      FacetsConfig config, TaxonomyReader taxo) {
    List<Map.Entry<String, FacetBuilder>> b = new ArrayList<>();
    b.add(Map.entry("author", new TaxonomyFacetBuilder(config, taxo, "Author")));
    b.add(Map.entry("author_top3", new TaxonomyFacetBuilder(config, taxo, "Author").withTopN(3)));
    b.add(Map.entry("date", new TaxonomyFacetBuilder(config, taxo, "Publish Date")));
    b.add(
        Map.entry(
            "date_2001", new TaxonomyFacetBuilder(config, taxo, "Publish Date", "2001").withTopN(5)));
    b.add(Map.entry("tag", new TaxonomyFacetBuilder(config, taxo, "Tag")));
    b.add(Map.entry("rating", new TaxonomyFacetBuilder(config, taxo, "Rating")));
    b.add(Map.entry("color", new TaxonomyFacetBuilder(config, taxo, "Color")));
    b.add(Map.entry("color_c1", new TaxonomyFacetBuilder(config, taxo, "Color", "c1")));
    b.add(Map.entry("price", new LongValueFacetBuilder("price")));
    b.add(Map.entry("price_count_top4", new LongValueFacetBuilder("price").withSortByCount().withTopN(4)));
    b.add(Map.entry("single", new LongValueFacetBuilder("single").withTopN(6)));
    b.add(Map.entry("skp_count", new LongValueFacetBuilder("skp").withSortByCount().withTopN(5)));
    b.add(Map.entry("range_disjoint", RangeFacetBuilderFactory.forLongRanges("price", disjointRanges())));
    b.add(
        Map.entry(
            "range_overlap", RangeFacetBuilderFactory.forLongRanges("price", overlappingRanges())));
    b.add(
        Map.entry(
            "range_single_disjoint",
            RangeFacetBuilderFactory.forLongRanges(
                "single_dim",
                MultiLongValuesSource.fromSingleValued(LongValuesSource.fromLongField("single")),
                disjointRanges())));
    b.add(
        Map.entry(
            "range_single_overlap",
            RangeFacetBuilderFactory.forLongRanges(
                "single_dim",
                MultiLongValuesSource.fromSingleValued(LongValuesSource.fromLongField("single")),
                overlappingRanges())));
    b.add(
        Map.entry(
            "range_multi_on_single",
            RangeFacetBuilderFactory.forLongRanges("single", overlappingRanges()).withTopN(3)));
    b.add(Map.entry("range_none", RangeFacetBuilderFactory.forLongRanges("price")));
    b.add(Map.entry("double_multi", RangeFacetBuilderFactory.forDoubleRanges("weight", doubleRanges())));
    b.add(
        Map.entry(
            "double_single",
            RangeFacetBuilderFactory.forDoubleRanges(
                "dweight",
                MultiDoubleValuesSource.fromSingleValued(DoubleValuesSource.fromDoubleField("dweight")),
                doubleRanges())));
    LongValueFacetCutter common = new LongValueFacetCutter("single");
    b.add(Map.entry("common", new CommonFacetBuilder("custom", common, common).withTopN(4)));
    return b;
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("sandbox_facet");
    if (Files.exists(root)) IOUtils.rm(root);
    Path indexDir = root.resolve("index");
    Path taxoDir = root.resolve("taxo");
    Files.createDirectories(indexDir);
    Files.createDirectories(taxoDir);
    FacetsConfig config = config();
    try (Directory id = FSDirectory.open(indexDir);
        Directory td = FSDirectory.open(taxoDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      IndexWriter w = new IndexWriter(id, cfg);
      DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(td);
      int i = 0;
      for (int size : SEGMENTS) {
        for (int k = 0; k < size; k++, i++) w.addDocument(config.build(tw, doc(i)));
        w.flush();
      }
      w.deleteDocuments(new Term("del", "y"));
      w.commit();
      w.close();
      tw.close();
    }

    StringBuilder builders = new StringBuilder();
    StringBuilder sideways = new StringBuilder();
    StringBuilder recorders = new StringBuilder();
    StringBuilder post = new StringBuilder();
    StringBuilder histograms = new StringBuilder();
    StringBuilder errors = new StringBuilder();
    try (Directory id = FSDirectory.open(indexDir);
        Directory td = FSDirectory.open(taxoDir);
        DirectoryReader reader = DirectoryReader.open(id);
        DirectoryTaxonomyReader taxo = new DirectoryTaxonomyReader(td)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      if (searcher.getSlices().length != 1) throw new AssertionError("one slice expected");
      if (reader.leaves().size() != SEGMENTS.length) throw new AssertionError("segments");

      for (Map.Entry<String, Query> base : bases().entrySet()) {
        // Every builder through one orchestrator: builders of one key share a collector.
        List<Map.Entry<String, FacetBuilder>> bs = builders(config, taxo);
        FacetOrchestrator orchestrator = new FacetOrchestrator();
        for (Map.Entry<String, FacetBuilder> e : bs) orchestrator.addBuilder(e.getValue());
        Integer hits =
            orchestrator.collect(base.getValue(), searcher, new TotalHitCountCollectorManager(searcher.getSlices()));
        builders.append(base.getKey()).append("\t#hits\t").append(hits).append('\n');
        for (Map.Entry<String, FacetBuilder> e : bs) {
          builders.append(base.getKey()).append('\t').append(e.getKey()).append('\t');
          builders.append(render(e.getValue().getResult())).append('\n');
        }
        recorders(base.getKey(), base.getValue(), searcher, config, taxo, recorders);
        post(base.getKey(), base.getValue(), searcher, config, taxo, post);
      }

      // Drill sideways.
      String[][][] drills = {
        {},
        {{"Author", "a3"}},
        {{"Author", "a3"}, {"Tag", "t1"}},
        {{"Tag", "t2"}, {"Author", "a5"}},
        {{"Publish Date", "2001"}},
      };
      String[] drillNames = {"none", "author", "author_tag", "tag_author", "date"};
      for (int d = 0; d < drills.length; d++) {
        for (String baseName : new String[] {"all", "even"}) {
          Query base = bases().get(baseName);
          DrillDownQuery ddq = new DrillDownQuery(config, base);
          for (String[] step : drills[d]) ddq.add(step[0], Arrays.copyOfRange(step, 1, step.length));
          DrillSidewaysFacetOrchestrator o = new DrillSidewaysFacetOrchestrator();
          Map<String, FacetBuilder> named = new LinkedHashMap<>();
          for (String[] step : drills[d]) {
            FacetBuilder fb = new TaxonomyFacetBuilder(config, taxo, step[0]);
            named.put("sideways:" + step[0], fb);
            o.addDrillSidewaysBuilder(step[0], fb);
          }
          FacetBuilder dateB = new TaxonomyFacetBuilder(config, taxo, "Publish Date").withTopN(2);
          FacetBuilder priceB = new LongValueFacetBuilder("price").withTopN(3);
          FacetBuilder rangeB = RangeFacetBuilderFactory.forLongRanges("price", disjointRanges());
          named.put("down:date", dateB);
          named.put("down:price", priceB);
          named.put("down:range", rangeB);
          o.addDrillDownBuilder(dateB).addDrillDownBuilder(priceB).addDrillDownBuilder(rangeB);
          String key = drillNames[d] + "/" + baseName;
          // collect returns the drill-down MultiCollectorManager's Object[], not the main
          // collector's result (an unchecked cast): its first element is the hit count.
          Object hits =
              ((Object[])
                      (Object)
                          o.collect(
                              ddq,
                              new DrillSideways(searcher, config, taxo),
                              new TotalHitCountCollectorManager(searcher.getSlices())))
                  [0];
          sideways.append(key).append("\t#hits\t").append(hits).append('\n');
          for (Map.Entry<String, FacetBuilder> e : named.entrySet()) {
            sideways.append(key).append('\t').append(e.getKey()).append('\t');
            sideways.append(render(e.getValue().getResult())).append('\n');
          }
        }
      }
      // A sideways dim the query does not drill on: one manager too many.
      try {
        DrillDownQuery ddq = new DrillDownQuery(config);
        ddq.add("Author", "a3");
        DrillSidewaysFacetOrchestrator o = new DrillSidewaysFacetOrchestrator();
        o.addDrillSidewaysBuilder("Author", new TaxonomyFacetBuilder(config, taxo, "Author"));
        o.addDrillSidewaysBuilder("Tag", new TaxonomyFacetBuilder(config, taxo, "Tag"));
        o.collect(ddq, new DrillSideways(searcher, config, taxo));
        errors.append("sideways_extra_dim\tnone\n");
      } catch (RuntimeException e) {
        errors.append("sideways_extra_dim\t").append(err(e)).append('\n');
      }
      // Two builders of different key for one sideways dim.
      try {
        DrillDownQuery ddq = new DrillDownQuery(config);
        ddq.add("Author", "a3");
        DrillSidewaysFacetOrchestrator o = new DrillSidewaysFacetOrchestrator();
        o.addDrillSidewaysBuilder("Author", new TaxonomyFacetBuilder(config, taxo, "Author"));
        o.addDrillSidewaysBuilder("Author", new LongValueFacetBuilder("price"));
        o.collect(ddq, new DrillSideways(searcher, config, taxo));
        errors.append("sideways_two_managers\tnone\n");
      } catch (RuntimeException e) {
        errors.append("sideways_two_managers\t").append(err(e)).append('\n');
      }

      histograms(searcher, histograms, errors);
      errors(config, taxo, errors);
    }

    Files.writeString(root.resolve("builders.tsv"), builders.toString(), StandardCharsets.UTF_8);
    Files.writeString(root.resolve("sideways.tsv"), sideways.toString(), StandardCharsets.UTF_8);
    Files.writeString(root.resolve("recorders.tsv"), recorders.toString(), StandardCharsets.UTF_8);
    Files.writeString(root.resolve("post.tsv"), post.toString(), StandardCharsets.UTF_8);
    Files.writeString(root.resolve("histograms.tsv"), histograms.toString(), StandardCharsets.UTF_8);
    Files.writeString(root.resolve("errors.tsv"), errors.toString(), StandardCharsets.UTF_8);
  }

  static String ords(int[] ords) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < ords.length; i++) {
      if (i > 0) sb.append(',');
      sb.append(ords[i]);
    }
    return sb.toString();
  }

  static String path(FacetLabel l) {
    return l == null ? "null" : esc(String.join("/", l.components));
  }

  static void recorders(
      String q,
      Query query,
      IndexSearcher searcher,
      FacetsConfig config,
      TaxonomyReader taxo,
      StringBuilder out)
      throws IOException {
    // Taxonomy counts with sum and max aggregations of "single" and "pt".
    TaxonomyFacetsCutter cutter = new TaxonomyFacetsCutter("$facets", config, taxo);
    CountFacetRecorder count = new CountFacetRecorder();
    LongAggregationsFacetRecorder agg =
        new LongAggregationsFacetRecorder(
            new LongValuesSource[] {
              LongValuesSource.fromLongField("single"),
              LongValuesSource.fromLongField("single"),
              LongValuesSource.fromLongField("skp")
            },
            new Reducer[] {Reducer.SUM, Reducer.MAX, Reducer.SUM});
    MultiFacetsRecorder multi = new MultiFacetsRecorder(count, agg);
    searcher.search(query, new FacetFieldCollectorManager<>(cutter, multi));
    TaxonomyOrdLabelBiMap labels = new TaxonomyOrdLabelBiMap(taxo);
    out.append(q).append("\ttaxo_empty\t").append(count.isEmpty()).append('\n');
    int[] all = count.recordedOrds().toArray();
    Arrays.sort(all);
    StringBuilder sb = new StringBuilder();
    for (int ord : all) {
      sb.append(path(labels.getLabel(ord))).append('=').append(count.getCount(ord));
      sb.append('/').append(agg.getRecordedValue(ord, 0));
      sb.append('/').append(agg.getRecordedValue(ord, 1));
      sb.append('/').append(agg.getRecordedValue(ord, 2)).append(',');
    }
    out.append(q).append("\ttaxo_all\t").append(sb).append('\n');
    for (String dim : new String[] {"Author", "Tag", "Publish Date", "Rating"}) {
      int dimOrd = labels.getOrd(new FacetLabel(dim));
      for (int agid = 0; agid < 3; agid++) {
        int[] top =
            new TopnOrdinalIterator<>(
                    new TaxonomyChildrenOrdinalIterator(
                        count.recordedOrds(), taxo.getParallelTaxonomyArrays().parents(), dimOrd),
                    ComparableUtils.byAggregatedValue(count, agg, agid),
                    4)
                .toArray();
        StringBuilder t = new StringBuilder();
        for (FacetLabel l : labels.getLabels(top)) t.append(path(l)).append(',');
        out.append(q).append("\ttop_agg").append(agid).append(':').append(dim).append('\t');
        out.append(t).append('\n');
      }
      int[] children =
          new TaxonomyChildrenOrdinalIterator(
                  count.recordedOrds(), taxo.getParallelTaxonomyArrays().parents(), dimOrd)
              .toArray();
      ComparableUtils.sort(children, ComparableUtils.byOrdinal());
      StringBuilder t = new StringBuilder();
      for (FacetLabel l : labels.getLabels(children)) t.append(path(l)).append(',');
      out.append(q).append("\tby_ordinal:").append(dim).append('\t').append(t).append('\n');
    }
    FacetLabel[] candidates = {
      new FacetLabel("Author", "a1"),
      new FacetLabel("Author", "zz"),
      new FacetLabel("Tag", "t3"),
      new FacetLabel("Author", "a1"),
      new FacetLabel("Nope"),
    };
    int[] cand = new CandidateSetOrdinalIterator(count, candidates, labels).toArray();
    StringBuilder cs = new StringBuilder();
    for (int ord : cand) cs.append(path(labels.getLabel(ord))).append('=').append(count.getCount(ord)).append(',');
    out.append(q).append("\tcandidates\t").append(cs).append('\n');

    // No rollup: a dim's own count is only what documents recorded on it.
    TaxonomyFacetsCutter noRollup = new TaxonomyFacetsCutter("$facets", config, taxo, true);
    CountFacetRecorder plain = new CountFacetRecorder();
    searcher.search(query, new FacetFieldCollectorManager<>(noRollup, plain));
    StringBuilder nr = new StringBuilder();
    for (String dim : new String[] {"Author", "Publish Date", "Rating", "Tag"}) {
      int dimOrd = labels.getOrd(new FacetLabel(dim));
      nr.append(dim).append('=').append(plain.isEmpty() ? -1 : plain.getCount(dimOrd)).append(',');
    }
    int y2001 = labels.getOrd(new FacetLabel("Publish Date", "2001"));
    nr.append("2001=").append(plain.isEmpty() ? -1 : plain.getCount(y2001));
    out.append(q).append("\tno_rollup\t").append(nr).append('\n');

    // Long values: ordinals, values, counts, and the two value sorts.
    LongValueFacetCutter lv = new LongValueFacetCutter("price");
    CountFacetRecorder lvCount = new CountFacetRecorder();
    LongAggregationsFacetRecorder lvAgg =
        new LongAggregationsFacetRecorder(
            new LongValuesSource[] {LongValuesSource.fromLongField("pt")},
            new Reducer[] {Reducer.MAX});
    searcher.search(query, new FacetFieldCollectorManager<>(lv, new MultiFacetsRecorder(lvCount, lvAgg)));
    int[] lvOrds = lvCount.recordedOrds().toArray();
    Arrays.sort(lvOrds);
    StringBuilder lvs = new StringBuilder();
    for (int ord : lvOrds) {
      lvs.append(ord).append(':').append(lv.getValue(ord)).append('=').append(lvCount.getCount(ord));
      lvs.append('/').append(lvAgg.getRecordedValue(ord, 0)).append(',');
    }
    out.append(q).append("\tlong_ords\t").append(lvs).append('\n');
    int[] byCount = lvCount.recordedOrds().toArray();
    ComparableUtils.sort(byCount, ComparableUtils.byCount(lvCount, lv));
    StringBuilder bc = new StringBuilder();
    for (FacetLabel l : lv.getLabels(byCount)) bc.append(path(l)).append(',');
    out.append(q).append("\tlong_by_count_value\t").append(bc).append('\n');
    int[] byValue = lvCount.recordedOrds().toArray();
    ComparableUtils.sort(byValue, ComparableUtils.byLongValue(lv));
    StringBuilder bv = new StringBuilder();
    for (int ord : byValue) bv.append(lv.getValue(ord)).append(',');
    out.append(q).append("\tlong_by_value\t").append(bv).append('\n');

    // Range labels.
    RangeOrdToLabel rl = new RangeOrdToLabel(disjointRanges());
    out.append(q).append("\trange_labels\t");
    for (FacetLabel l : rl.getLabels(new int[] {0, 3, -1, 4})) out.append(path(l)).append(',');
    out.append('\n');
  }

  static void post(
      String q,
      Query query,
      IndexSearcher searcher,
      FacetsConfig config,
      TaxonomyReader taxo,
      StringBuilder out)
      throws IOException {
    FacetsCollector fc = searcher.search(query, new FacetsCollectorManager());
    TaxonomyFacetsCutter cutter = new TaxonomyFacetsCutter("$facets", config, taxo);
    CountFacetRecorder down = new CountFacetRecorder();
    PostCollectionFaceting<FacetFieldCollector, CountFacetRecorder, FacetFieldCollector, CountFacetRecorder> p =
        new PostCollectionFaceting<>(new FacetFieldCollectorManager<>(cutter, down), fc, null);
    PostCollectionFaceting.Result<CountFacetRecorder, CountFacetRecorder> r = p.collect();
    TaxonomyOrdLabelBiMap labels = new TaxonomyOrdLabelBiMap(taxo);
    out.append(q).append("\tdown_only\t").append(counts(r.drillDownResult(), labels)).append('\n');

    // Drill sideways collectors: the base query drilled on Author a3 and Tag t1.
    DrillDownQuery ddq = new DrillDownQuery(config, query);
    ddq.add("Author", "a3");
    ddq.add("Tag", "t1");
    DrillSideways ds = new DrillSideways(searcher, config, taxo);
    DrillSideways.Result<FacetsCollector, FacetsCollector> dsr =
        ds.search(
            ddq,
            new FacetsCollectorManager(),
            List.of(new FacetsCollectorManager(), new FacetsCollectorManager()));
    Map<String, FacetFieldCollectorManager<CountFacetRecorder>> managers = new TreeMap<>();
    Map<String, CountFacetRecorder> sidewaysRecorders = new TreeMap<>();
    for (String dim : new String[] {"Author", "Tag", "Unused"}) {
      CountFacetRecorder rec = new CountFacetRecorder();
      sidewaysRecorders.put(dim, rec);
      managers.put(dim, new FacetFieldCollectorManager<>(new TaxonomyFacetsCutter("$facets", config, taxo), rec));
    }
    Map<String, FacetsCollector> sidewaysCollectors = new TreeMap<>();
    sidewaysCollectors.put("Author", dsr.drillSidewaysResults().get(0));
    sidewaysCollectors.put("Tag", dsr.drillSidewaysResults().get(1));
    CountFacetRecorder dd = new CountFacetRecorder();
    PostCollectionFaceting<FacetFieldCollector, CountFacetRecorder, FacetFieldCollector, CountFacetRecorder> ps =
        new PostCollectionFaceting<>(
            new FacetFieldCollectorManager<>(new TaxonomyFacetsCutter("$facets", config, taxo), dd),
            managers,
            dsr.drillDownResult(),
            sidewaysCollectors,
            null);
    PostCollectionFaceting.Result<CountFacetRecorder, CountFacetRecorder> rs = ps.collect();
    out.append(q).append("\tsideways_down\t").append(counts(rs.drillDownResult(), labels)).append('\n');
    for (String dim : new TreeMap<>(rs.drillSidewaysResults()).keySet()) {
      out.append(q).append("\tsideways:").append(dim).append('\t');
      out.append(counts(rs.drillSidewaysResults().get(dim), labels)).append('\n');
    }
    out.append(q).append("\tsideways_dims\t").append(new TreeMap<>(rs.drillSidewaysResults()).keySet()).append('\n');
    // Sideways only, no drill-down manager.
    CountFacetRecorder onlyAuthor = new CountFacetRecorder();
    Map<String, FacetFieldCollectorManager<CountFacetRecorder>> one = new TreeMap<>();
    one.put("Author", new FacetFieldCollectorManager<>(new TaxonomyFacetsCutter("$facets", config, taxo), onlyAuthor));
    PostCollectionFaceting<FacetFieldCollector, CountFacetRecorder, FacetFieldCollector, CountFacetRecorder> pn =
        new PostCollectionFaceting<>(null, one, dsr.drillDownResult(), sidewaysCollectors, null);
    PostCollectionFaceting.Result<CountFacetRecorder, CountFacetRecorder> rn = pn.collect();
    out.append(q).append("\tsideways_only\t").append(rn.drillDownResult()).append(' ');
    out.append(counts(rn.drillSidewaysResults().get("Author"), labels)).append('\n');
  }

  static String counts(CountFacetRecorder rec, TaxonomyOrdLabelBiMap labels) throws IOException {
    if (rec.isEmpty()) return "empty";
    int[] ords = rec.recordedOrds().toArray();
    Arrays.sort(ords);
    StringBuilder sb = new StringBuilder();
    for (int ord : ords) sb.append(path(labels.getLabel(ord))).append('=').append(rec.getCount(ord)).append(',');
    return sb.toString();
  }

  static void histograms(IndexSearcher searcher, StringBuilder out, StringBuilder errors)
      throws IOException {
    Map<String, Query> queries = new LinkedHashMap<>();
    queries.put("all", new MatchAllDocsQuery());
    queries.put("even", new TermQuery(new Term("q", "even")));
    queries.put("rare", new TermQuery(new Term("q", "rare")));
    queries.put("pt_range", LongPoint.newRangeQuery("pt", 100, 700));
    queries.put("pt_range_wrapped", new ConstantScoreQuery(new BoostQuery(LongPoint.newRangeQuery("pt", -50, 333), 2f)));
    queries.put("pt_range_empty", LongPoint.newRangeQuery("pt", 5000, 6000));
    queries.put("ipt_range", IntPoint.newRangeQuery("ipt", -100, 150));
    // A lone SHOULD clause: rewritten to its range, so the bulk path.
    queries.put("pt_range_should", new BooleanQuery.Builder()
        .add(LongPoint.newRangeQuery("pt", 20, 900), BooleanClause.Occur.SHOULD).build());
    Object[][] specs = {
      {"pt", 2L, 1024}, {"pt", 7L, 1024}, {"pt", 100L, 1024}, {"pt", 500L, 1024},
      {"pt", 1000L, 1024}, {"pt", 100000L, 1024}, {"pt", 3L, 100},
      {"ipt", 50L, 1024}, {"ipt", 300L, 1024}, {"ipt", 1000L, 1024},
      {"skp", 10L, 1024}, {"skp", 1000L, 1024}, {"skp", 2L, 1024},
      {"price", 5L, 1024}, {"price", 2L, 20}, {"single", 64L, 1024},
      {"missing", 10L, 1024}, {"kw", 10L, 1024},
    };
    for (Map.Entry<String, Query> q : queries.entrySet()) {
      for (Object[] s : specs) {
        String field = (String) s[0];
        long width = (Long) s[1];
        int max = (Integer) s[2];
        out.append(q.getKey()).append('\t').append(field).append('\t').append(width).append('\t').append(max).append('\t');
        try {
          LongIntHashMap m = searcher.search(q.getValue(), new HistogramCollectorManager(field, width, max));
          TreeMap<Long, Integer> sorted = new TreeMap<>();
          for (var c : m) sorted.put(c.key, c.value);
          StringBuilder sb = new StringBuilder();
          for (Map.Entry<Long, Integer> e : sorted.entrySet()) sb.append(e.getKey()).append('=').append(e.getValue()).append(',');
          out.append(sb);
        } catch (RuntimeException e) {
          out.append("error ").append(err(e));
        }
        out.append('\n');
      }
    }
    for (Object[] bad : new Object[][] {{"pt", 1L, 10}, {"pt", 10L, 0}}) {
      try {
        new HistogramCollectorManager((String) bad[0], (Long) bad[1], (Integer) bad[2]);
        errors.append("histogram ").append(bad[1]).append(' ').append(bad[2]).append("\tnone\n");
      } catch (RuntimeException e) {
        errors.append("histogram ").append(bad[1]).append(' ').append(bad[2]).append('\t').append(err(e)).append('\n');
      }
    }
  }

  static void errors(FacetsConfig config, TaxonomyReader taxo, StringBuilder out) throws IOException {
    try {
      new TaxonomyFacetBuilder(config, taxo, "Unconfigured");
      out.append("unconfigured_dim\tnone\n");
    } catch (RuntimeException e) {
      out.append("unconfigured_dim\t").append(err(e)).append('\n');
    }
    try {
      new TopnOrdinalIterator<>(OrdinalIterator.EMPTY, ComparableUtils.byOrdinal(), 0);
      out.append("topn_zero\tnone\n");
    } catch (RuntimeException e) {
      out.append("topn_zero\t").append(err(e)).append('\n');
    }
    LengthOrdinalIterator li = new LengthOrdinalIterator(OrdinalIterator.fromArray(new int[] {3, -1, 5, -1}));
    try {
      li.length();
      out.append("length_early\tnone\n");
    } catch (RuntimeException e) {
      out.append("length_early\t").append(err(e)).append('\n');
    }
    out.append("from_array\t").append(ords(li.toArray())).append(' ').append(li.length()).append('\n');
    LongAggregationsFacetRecorder agg =
        new LongAggregationsFacetRecorder(
            new LongValuesSource[] {LongValuesSource.fromLongField("single")}, new Reducer[] {Reducer.SUM});
    for (int idx : new int[] {-1, 1}) {
      try {
        agg.getRecordedValue(0, idx);
        out.append("agg_value_id ").append(idx).append("\tnone\n");
      } catch (RuntimeException e) {
        out.append("agg_value_id ").append(idx).append('\t').append(err(e)).append('\n');
      }
    }
    MultiFacetsRecorder multi = new MultiFacetsRecorder();
    try {
      multi.recordedOrds();
    } catch (RuntimeException e) {
      out.append("multi_recorded_ords\t").append(err(e)).append('\n');
    }
    try {
      multi.isEmpty();
    } catch (RuntimeException e) {
      out.append("multi_is_empty\t").append(err(e)).append('\n');
    }
    try {
      multi.contains(1);
    } catch (RuntimeException e) {
      out.append("multi_contains\t").append(err(e)).append('\n');
    }
    for (long[] ab : new long[][] {{Long.MAX_VALUE, 1}, {Integer.MAX_VALUE, 1}}) {
      try {
        out.append("sum ").append(ab[0]).append("\t");
        out.append(Reducer.SUM.reduce(ab[0], ab[1])).append(' ');
        out.append(Reducer.SUM.reduce((int) ab[0], (int) ab[1])).append('\n');
      } catch (ArithmeticException e) {
        out.append(err(e)).append('\n');
      }
    }
    out.append("max\t").append(Reducer.MAX.reduce(-3L, 7L)).append(' ').append(Reducer.MAX.reduce(2.5, -1.0))
        .append(' ').append(Reducer.MAX.reduce(Float.NaN, 1f)).append(' ').append(Reducer.SUM.reduce(0.1, 0.2)).append('\n');
  }
}
