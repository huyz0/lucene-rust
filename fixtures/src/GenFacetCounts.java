import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.facet.FacetField;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.Facets;
import org.apache.lucene.facet.FacetsCollector;
import org.apache.lucene.facet.FacetsCollectorManager;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.LabelAndValue;
import org.apache.lucene.facet.MultiFacets;
import org.apache.lucene.facet.taxonomy.AssociationAggregationFunction;
import org.apache.lucene.facet.taxonomy.FacetLabel;
import org.apache.lucene.facet.taxonomy.FastTaxonomyFacetCounts;
import org.apache.lucene.facet.taxonomy.FloatAssociationFacetField;
import org.apache.lucene.facet.taxonomy.IntAssociationFacetField;
import org.apache.lucene.facet.taxonomy.TaxonomyFacetFloatAssociations;
import org.apache.lucene.facet.taxonomy.TaxonomyFacetIntAssociations;
import org.apache.lucene.facet.taxonomy.TaxonomyFacetLabels;
import org.apache.lucene.facet.taxonomy.TaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyWriter;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IOUtils;

/**
 * M13 T13.4b: taxonomy facet counting. Writes {@code facet_counts/}:
 *
 * <ul>
 *   <li>{@code index/} and {@code taxo/}: {@link #DOCS} documents in four segments (deletions in
 *       each) plus a fifth of {@link #BARE} documents without facet fields, built through {@link
 *       #config} -- flat, hierarchical, multi-valued, {@code requireDimCount} and custom index
 *       field dims, int and float associations, more than 1024 categories (so a selective query
 *       counts sparsely).
 *   <li>{@code calls.tsv}: every {@code Facets} call over every facets object of every query (see
 *       {@link #probe}): {@code facets, query, method, n1, n2, dim|path..., result}.
 *   <li>{@code search.tsv}: {@code FacetsCollectorManager.search}/{@code searchAfter} top docs.
 *   <li>{@code labels.tsv}: {@code TaxonomyFacetLabels} over sampled documents of every leaf.
 *   <li>{@code build.tsv}: {@code FacetsConfig.build} of the first documents (field, kind,
 *       value; binary values in hex); {@code errors.tsv}: refusals and their messages.
 * </ul>
 *
 * A result is {@code null}, {@code err<TAB>message}, {@code num<TAB>value}, {@code res<TAB>...}
 * (see {@link #res}) or {@code list<TAB>n} followed by each result's own columns after a {@code
 * ||} column. A number is {@code i<int>} or {@code f<float bits, hex>}. {@code ress} marks a
 * {@code getAllChildren} counted sparsely: its labels are sorted, as Lucene's hash-table order is
 * not reproducible. Float facets are not run over the sparse query (their sums would follow that
 * order too), except the order-free {@code MAX}.
 */
public class GenFacetCounts {
  static final int DOCS = 2400;
  static final int PER_SEGMENT = 600;
  static final int BARE = 30;

  static FacetsConfig config() {
    FacetsConfig c = new FacetsConfig();
    c.setHierarchical("Publish Date", true);
    c.setMultiValued("Tag", true);
    c.setMultiValued("Cat", true);
    c.setRequireDimCount("Cat", true);
    c.setHierarchical("Path", true);
    c.setMultiValued("Path", true);
    c.setRequireDimCount("Path", true);
    c.setHierarchical("Loc", true);
    c.setMultiValued("Loc", true);
    c.setIndexFieldName("Color", "$color");
    c.setHierarchical("Color", true);
    c.setIndexFieldName("IntA", "$int_assoc");
    c.setMultiValued("IntA", true);
    c.setIndexFieldName("IntB", "$int_assoc");
    c.setIndexFieldName("FloatA", "$float_assoc");
    c.setMultiValued("FloatA", true);
    c.setIndexFieldName("FloatB", "$float_assoc");
    return c;
  }

  static Document doc(int i) {
    Document d = new Document();
    d.add(new StringField("q", "all", Field.Store.NO));
    if (i % 2 == 0) d.add(new StringField("q", "even", Field.Store.NO));
    if (i % 37 == 5) d.add(new StringField("q", "rare", Field.Store.NO));
    if (i % 11 == 3) d.add(new StringField("del", "y", Field.Store.NO));
    if (i % 10 != 0) d.add(new NumericDocValuesField("num", i % 17));
    d.add(new FacetField("Author", "a" + (i * 7 % 23)));
    d.add(new FacetField("Publish Date", "" + (2000 + i % 5), "" + (1 + i % 12), "" + (1 + i % 28)));
    d.add(new FacetField("Tag", "t" + (i % 9)));
    d.add(new FacetField("Tag", "t" + (i % 7 + 2)));
    if (i % 3 != 0) {
      d.add(new FacetField("Cat", "c" + (i % 4)));
      d.add(new FacetField("Cat", "c" + (i % 6)));
    }
    d.add(new FacetField("Path", "p" + (i % 3), "q" + (i % 5)));
    d.add(new FacetField("Path", "p" + (i % 2)));
    if (i % 4 != 1) d.add(new FacetField("Loc", "l" + (i % 4), "m" + (i % 3)));
    d.add(new FacetField("Many", "m" + (i % 1100)));
    d.add(new FacetField("Color", "c" + (i % 5), "s" + (i % 3)));
    d.add(new IntAssociationFacetField(i % 10 - 3, "IntA", "x" + (i % 6)));
    d.add(new IntAssociationFacetField((i * 3) % 10 - 4, "IntA", "x" + ((i + 1) % 6)));
    d.add(new IntAssociationFacetField(i % 7 + 1, "IntB", "z" + (i % 4)));
    d.add(new FloatAssociationFacetField((i % 13) * 0.37f - 1.1f, "FloatA", "y" + (i % 5)));
    d.add(new FloatAssociationFacetField(i * 0.013f, "FloatA", "y" + ((i + 2) % 5)));
    d.add(new FloatAssociationFacetField(i * 0.013f + 0.5f, "FloatB", "w" + (i % 3)));
    return d;
  }

  static Document bare(int i) {
    Document d = new Document();
    d.add(new StringField("q", "all", Field.Store.NO));
    if (i % 2 == 0) d.add(new StringField("q", "even", Field.Store.NO));
    return d;
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("facet_counts");
    if (Files.exists(root)) IOUtils.rm(root);
    Path indexDir = root.resolve("index");
    Path taxoDir = root.resolve("taxo");
    Files.createDirectories(indexDir);
    Files.createDirectories(taxoDir);
    FacetsConfig config = config();
    StringBuilder build = new StringBuilder();
    try (Directory id = FSDirectory.open(indexDir);
        Directory td = FSDirectory.open(taxoDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      IndexWriter w = new IndexWriter(id, cfg);
      DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(td);
      for (int i = 0; i < DOCS; i++) {
        Document built = config.build(tw, doc(i));
        if (i < 3) dumpBuilt(build, i, built);
        w.addDocument(built);
        if ((i + 1) % PER_SEGMENT == 0) w.flush();
      }
      for (int i = 0; i < BARE; i++) w.addDocument(config.build(tw, bare(i)));
      w.flush();
      w.deleteDocuments(new Term("del", "y"));
      w.commit();
      w.close();
      tw.close();
    }
    write(root.resolve("build.tsv"), build.toString());

    StringBuilder calls = new StringBuilder();
    StringBuilder search = new StringBuilder();
    StringBuilder labels = new StringBuilder();
    try (Directory id = FSDirectory.open(indexDir);
        Directory td = FSDirectory.open(taxoDir);
        DirectoryReader reader = DirectoryReader.open(id);
        DirectoryTaxonomyReader taxo = new DirectoryTaxonomyReader(td)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      if (searcher.getSlices().length != 1) throw new AssertionError("one slice expected");
      Map<String, Query> queries = new LinkedHashMap<>();
      queries.put("all", new MatchAllDocsQuery());
      queries.put("even", new TermQuery(new Term("q", "even")));
      queries.put("rare", new TermQuery(new Term("q", "rare")));
      queries.put("none", new TermQuery(new Term("q", "none")));
      for (Map.Entry<String, Query> e : queries.entrySet()) {
        String qn = e.getKey();
        Query q = e.getValue();
        FacetsCollector fc = searcher.search(q, new FacetsCollectorManager());
        FacetsCollector fcs = searcher.search(q, new FacetsCollectorManager(true));
        boolean sparse = taxo.getSize() >= 1024 && sparse(fc);
        Map<String, Facets> all = new LinkedHashMap<>();
        Facets counts = new FastTaxonomyFacetCounts(taxo, config, fc);
        Facets color = new FastTaxonomyFacetCounts("$color", taxo, config, fc);
        all.put("counts", counts);
        all.put("color", color);
        all.put("int_sum",
            new TaxonomyFacetIntAssociations("$int_assoc", taxo, config, fc, AssociationAggregationFunction.SUM));
        all.put("int_max",
            new TaxonomyFacetIntAssociations("$int_assoc", taxo, config, fc, AssociationAggregationFunction.MAX));
        if (!sparse) {
          all.put("float_sum", new TaxonomyFacetFloatAssociations(
              "$float_assoc", taxo, config, fc, AssociationAggregationFunction.SUM));
          all.put("vs_num", new TaxonomyFacetFloatAssociations(
              taxo, config, fc, AssociationAggregationFunction.SUM, DoubleValuesSource.fromLongField("num")));
        }
        all.put("float_max", new TaxonomyFacetFloatAssociations(
            "$float_assoc", taxo, config, fc, AssociationAggregationFunction.MAX));
        all.put("vs_score_max", new TaxonomyFacetFloatAssociations(
            taxo, config, fcs, AssociationAggregationFunction.MAX, DoubleValuesSource.SCORES));
        all.put("multi", new MultiFacets(Map.of("Color", color), counts));
        all.put("multi_nodefault", new MultiFacets(Map.of("Color", color)));
        if (qn.equals("all")) {
          all.put("count_all", new FastTaxonomyFacetCounts("$facets", reader, taxo, config));
          all.put("count_all_color", new FastTaxonomyFacetCounts("$color", reader, taxo, config));
        }
        for (Map.Entry<String, Facets> f : all.entrySet()) {
          probe(calls, f.getKey(), qn, f.getValue(), sparse && !f.getKey().startsWith("count_all"));
        }

        // FacetsCollectorManager.search / searchAfter.
        int[] ns = qn.equals("rare") || qn.equals("none") ? new int[] {0, 1, 7, 100000} : new int[] {0, 1, 7};
        for (int n : ns) {
          FacetsCollectorManager.FacetsResult r =
              FacetsCollectorManager.search(searcher, q, n, new FacetsCollectorManager());
          search.append(qn).append("\tsearch\t").append(n).append('\t').append(topDocs(r));
          search.append('\t').append(dims(new FastTaxonomyFacetCounts(taxo, config, r.facetsCollector())));
          search.append('\n');
        }
        ScoreDoc[] first = FacetsCollectorManager.search(searcher, q, 3, new FacetsCollectorManager())
            .topDocs().scoreDocs;
        if (first.length == 3) {
          FacetsCollectorManager.FacetsResult r = FacetsCollectorManager.searchAfter(
              searcher, first[2], q, 5, new FacetsCollectorManager());
          search.append(qn).append("\tafter\t").append(first[2].doc).append('\t')
              .append(Integer.toHexString(Float.floatToIntBits(first[2].score))).append('\t')
              .append(topDocs(r)).append('\n');
        }
      }
      try {
        FacetsCollectorManager.searchAfter(searcher, new ScoreDoc(reader.maxDoc(), 1f),
            new MatchAllDocsQuery(), 5, new FacetsCollectorManager());
        search.append("error\tnone\n");
      } catch (IllegalArgumentException ex) {
        search.append("error\t").append(ex.getMessage()).append('\n');
      }

      // TaxonomyFacetLabels.
      TaxonomyFacetLabels tfl = new TaxonomyFacetLabels(taxo, "$facets");
      TaxonomyFacetLabels tflColor = new TaxonomyFacetLabels(taxo, "$color");
      for (LeafReaderContext ctx : reader.leaves()) {
        TaxonomyFacetLabels.FacetLabelReader all = tfl.getFacetLabelReader(ctx);
        TaxonomyFacetLabels.FacetLabelReader byDim = tfl.getFacetLabelReader(ctx);
        TaxonomyFacetLabels.FacetLabelReader byDim2 = tfl.getFacetLabelReader(ctx);
        TaxonomyFacetLabels.FacetLabelReader color = tflColor.getFacetLabelReader(ctx);
        for (int doc = 0; doc < ctx.reader().maxDoc(); doc += 97) {
          labels.append(ctx.ord).append('\t').append(doc).append("\tall");
          for (FacetLabel l = all.nextFacetLabel(doc); l != null; l = all.nextFacetLabel(doc)) {
            labels.append('\t').append(String.join("/", l.components));
          }
          labels.append('\n');
          labels.append(ctx.ord).append('\t').append(doc).append("\tPublish Date");
          for (FacetLabel l = byDim.nextFacetLabel(doc, "Publish Date"); l != null;
              l = byDim.nextFacetLabel(doc, "Publish Date")) {
            labels.append('\t').append(String.join("/", l.components));
          }
          labels.append('\n');
          // One label of Tag, then whatever Path label follows it.
          labels.append(ctx.ord).append('\t').append(doc).append("\tTag+Path");
          FacetLabel t = byDim2.nextFacetLabel(doc, "Tag");
          labels.append('\t').append(t == null ? "null" : String.join("/", t.components));
          FacetLabel p = byDim2.nextFacetLabel(doc, "Path");
          labels.append('\t').append(p == null ? "null" : String.join("/", p.components));
          labels.append('\n');
          labels.append(ctx.ord).append('\t').append(doc).append("\tcolor");
          for (FacetLabel l = color.nextFacetLabel(doc); l != null; l = color.nextFacetLabel(doc)) {
            labels.append('\t').append(String.join("/", l.components));
          }
          labels.append('\n');
        }
      }
      StringBuilder errors = new StringBuilder();
      LeafReaderContext leaf0 = reader.leaves().get(0);
      error(errors, "labels_out_of_order", () -> {
        TaxonomyFacetLabels.FacetLabelReader r = tfl.getFacetLabelReader(leaf0);
        r.nextFacetLabel(5);
        r.nextFacetLabel(4);
      });
      error(errors, "labels_out_of_order_dim", () -> {
        TaxonomyFacetLabels.FacetLabelReader r = tfl.getFacetLabelReader(leaf0);
        r.nextFacetLabel(5, "Tag");
        r.nextFacetLabel(4, "Tag");
      });
      error(errors, "labels_missing_dim", () -> tfl.getFacetLabelReader(leaf0).nextFacetLabel(0, "Nope"));
      buildErrors(errors);
      write(root.resolve("errors.tsv"), errors.toString());
    }
    write(root.resolve("calls.tsv"), calls.toString());
    write(root.resolve("search.tsv"), search.toString());
    write(root.resolve("labels.tsv"), labels.toString());
  }

  static boolean sparse(FacetsCollector fc) {
    int maxDoc = 0;
    int hits = 0;
    for (FacetsCollector.MatchingDocs m : fc.getMatchingDocs()) {
      hits += m.totalHits();
      maxDoc += m.context().reader().maxDoc();
    }
    return hits < maxDoc / 10;
  }

  static final String[] DEFAULT_DIMS = {
    "Author", "Publish Date", "Tag", "Cat", "Path", "Loc", "Many", "Color", "IntA", "Nope"
  };
  static final String[] COLOR_DIMS = {"Color", "Author", "Nope"};
  static final String[] INT_DIMS = {"IntA", "IntB", "Author", "FloatA", "Nope"};
  static final String[] FLOAT_DIMS = {"FloatA", "FloatB", "IntB", "Nope"};

  /** Child paths probed under a dim (each also through {@code getSpecificValue}). */
  static String[][] paths(String dim) {
    return switch (dim) {
      case "Author" -> new String[][] {{"a3"}, {"a99"}};
      case "Publish Date" -> new String[][] {{"2001"}, {"2001", "3"}, {"2001", "3", "4"}, {"1900"}};
      case "Tag" -> new String[][] {{"t3"}};
      case "Cat" -> new String[][] {{"c1"}};
      case "Path" -> new String[][] {{"p1"}, {"p0"}};
      case "Loc" -> new String[][] {{"l2"}, {"l1"}};
      case "Many" -> new String[][] {{"m5"}};
      case "Color" -> new String[][] {{"c1"}, {"c9"}};
      case "IntA" -> new String[][] {{"x1"}};
      case "IntB" -> new String[][] {{"z2"}};
      case "FloatA" -> new String[][] {{"y1"}};
      case "FloatB" -> new String[][] {{"w0"}};
      default -> new String[][] {{"x"}};
    };
  }

  static void probe(StringBuilder sb, String name, String qn, Facets f, boolean sparse) {
    String[] dims = name.startsWith("color") || name.equals("count_all_color") ? COLOR_DIMS
        : name.startsWith("int") ? INT_DIMS
        : name.startsWith("float") ? FLOAT_DIMS
        : DEFAULT_DIMS;
    String pre = name + "\t" + qn + "\t";
    for (String dim : dims) {
      for (int n : new int[] {0, 1, 3, 50}) {
        line(sb, pre + "top\t" + n + "\t0\t" + dim, () -> res(f.getTopChildren(n, dim), false));
      }
      if (!dim.equals("Many") || qn.equals("rare")) {
        line(sb, pre + "all\t0\t0\t" + dim, () -> res(f.getAllChildren(dim), sparse));
      }
      line(sb, pre + "spec\t0\t0\t" + dim, () -> "num\t" + num(f.getSpecificValue(dim)));
      for (String[] p : paths(dim)) {
        String dp = dim + "|" + String.join("|", p);
        line(sb, pre + "top\t5\t0\t" + dp, () -> res(f.getTopChildren(5, dim, p), false));
        line(sb, pre + "all\t0\t0\t" + dp, () -> res(f.getAllChildren(dim, p), sparse));
        line(sb, pre + "spec\t0\t0\t" + dp, () -> "num\t" + num(f.getSpecificValue(dim, p)));
      }
    }
    for (int n : new int[] {0, 1, 4}) {
      line(sb, pre + "dims\t" + n + "\t0\t-", () -> list(f.getAllDims(n)));
    }
    for (int[] nn : new int[][] {{0, 1}, {1, 0}, {1, 1}, {2, 3}, {3, 2}, {100, 100}}) {
      line(sb, pre + "topdims\t" + nn[0] + "\t" + nn[1] + "\t-",
          () -> list(f.getTopDims(nn[0], nn[1])));
    }
  }

  interface Call {
    String run() throws Exception;
  }

  static void line(StringBuilder sb, String call, Call c) {
    sb.append(call).append('\t');
    try {
      sb.append(c.run());
    } catch (IllegalArgumentException e) {
      sb.append("err\t").append(e.getMessage());
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    sb.append('\n');
  }

  static String num(Number n) {
    if (n instanceof Integer) return "i" + n;
    if (n instanceof Float f) return "f" + Integer.toHexString(Float.floatToIntBits(f));
    throw new AssertionError(n.getClass().toString());
  }

  static String res(FacetResult r, boolean sorted) {
    if (r == null) return "null";
    LabelAndValue[] lv = r.labelValues.clone();
    if (sorted) Arrays.sort(lv, (a, b) -> a.label.compareTo(b.label));
    StringBuilder sb = new StringBuilder(sorted ? "ress" : "res");
    sb.append('\t').append(r.dim).append('\t').append(String.join("|", r.path)).append('\t')
        .append(num(r.value)).append('\t').append(r.childCount).append('\t');
    for (int i = 0; i < lv.length; i++) {
      if (i > 0) sb.append(',');
      sb.append(lv[i].label).append('=').append(num(lv[i].value)).append('=').append(lv[i].count);
    }
    return sb.toString();
  }

  /**
   * A list of results. {@code MultiFacets.getAllDims} adds a dim's {@code null} result to its
   * list; the Rust port has no null element and drops it, so they are dropped here too.
   */
  static String list(List<FacetResult> l) {
    List<FacetResult> nonNull = new ArrayList<>();
    for (FacetResult r : l) if (r != null) nonNull.add(r);
    StringBuilder sb = new StringBuilder("list\t").append(nonNull.size());
    for (FacetResult r : nonNull) sb.append("\t||\t").append(res(r, false));
    return sb.toString();
  }

  static String topDocs(FacetsCollectorManager.FacetsResult r) {
    StringBuilder sb = new StringBuilder();
    sb.append(r.topDocs().totalHits.value()).append('\t').append(r.topDocs().totalHits.relation())
        .append('\t');
    ScoreDoc[] sd = r.topDocs().scoreDocs;
    for (int i = 0; i < sd.length; i++) {
      if (i > 0) sb.append(',');
      sb.append(sd[i].doc).append(':').append(Integer.toHexString(Float.floatToIntBits(sd[i].score)));
    }
    return sb.toString();
  }

  /** {@code getAllDims(3)}'s dims and values. */
  static String dims(Facets f) throws Exception {
    StringBuilder sb = new StringBuilder();
    for (FacetResult r : f.getAllDims(3)) sb.append(r.dim).append('=').append(num(r.value)).append(';');
    return sb.toString();
  }

  static void dumpBuilt(StringBuilder sb, int i, Document built) {
    for (IndexableField f : built.getFields()) {
      sb.append(i).append('\t').append(f.name()).append('\t');
      if (f instanceof SortedNumericDocValuesField) {
        sb.append("sndv\t").append(f.numericValue());
      } else if (f instanceof BinaryDocValuesField) {
        sb.append("bdv\t").append(hex(f.binaryValue()));
      } else if (f instanceof NumericDocValuesField) {
        sb.append("ndv\t").append(f.numericValue());
      } else if (f instanceof StringField) {
        sb.append("string\t").append(f.stringValue());
      } else {
        throw new AssertionError(f.getClass().toString());
      }
      sb.append('\n');
    }
  }

  static String hex(BytesRef b) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < b.length; i++) sb.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    return sb.toString();
  }

  interface Thrower {
    void run() throws Exception;
  }

  static void error(StringBuilder sb, String name, Thrower t) {
    try {
      t.run();
      sb.append(name).append("\tnone\t\n");
    } catch (Exception e) {
      sb.append(name).append('\t').append(e.getClass().getSimpleName()).append('\t')
          .append(e.getMessage() == null ? "" : e.getMessage()).append('\n');
    }
  }

  static void buildErrors(StringBuilder sb) throws Exception {
    try (Directory d = new ByteBuffersDirectory();
        DirectoryTaxonomyWriter w = new DirectoryTaxonomyWriter(d)) {
      FacetsConfig c = new FacetsConfig();
      c.setHierarchical("H", true);
      c.setRequireDimCount("R", true);
      c.setMultiValued("R", true);
      error(sb, "assoc_hierarchical", () -> c.build(w, docOf(new IntAssociationFacetField(1, "H", "a"))));
      error(sb, "assoc_require_dim_count",
          () -> c.build(w, docOf(new IntAssociationFacetField(1, "R", "a"))));
      error(sb, "assoc_not_multivalued", () -> c.build(w, docOf(
          new IntAssociationFacetField(1, "A", "a"), new IntAssociationFacetField(2, "A", "b"))));
      error(sb, "assoc_first_int", () -> c.build(w, docOf(new IntAssociationFacetField(1, "A", "a"))));
      error(sb, "assoc_mix_float", () -> c.build(w, docOf(new FloatAssociationFacetField(1f, "B", "a"))));
      error(sb, "assoc_mix_float_again",
          () -> c.build(w, docOf(new FloatAssociationFacetField(1f, "B", "a"))));
      FacetsConfig c2 = new FacetsConfig();
      error(sb, "assoc_first_float",
          () -> c2.build(w, docOf(new FloatAssociationFacetField(1f, "B", "a"))));
      error(sb, "assoc_mix_int", () -> c2.build(w, docOf(new IntAssociationFacetField(1, "A", "a"))));
      error(sb, "assoc_no_taxo_writer",
          () -> new FacetsConfig().build(docOf(new IntAssociationFacetField(1, "A", "a"))));
      error(sb, "assoc_empty_path", () -> new IntAssociationFacetField(1, "A"));
      error(sb, "assoc_empty_label", () -> new IntAssociationFacetField(1, "A", ""));
      error(sb, "assoc_empty_dim", () -> new FloatAssociationFacetField(1f, "", "a"));
      sb.append("int_to_string\t").append(new IntAssociationFacetField(-7, "A", "a", "b")).append('\n');
      sb.append("float_to_string\t").append(new FloatAssociationFacetField(1.5f, "A", "a")).append('\n');
    }
  }

  static Document docOf(Field... fields) {
    Document d = new Document();
    for (Field f : fields) d.add(f);
    return d;
  }

  static void write(Path p, String s) throws Exception {
    Files.writeString(p, s, StandardCharsets.UTF_8);
  }
}
