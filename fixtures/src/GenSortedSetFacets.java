import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.PrimitiveIterator;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.facet.DrillDownQuery;
import org.apache.lucene.facet.DrillSideways;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.Facets;
import org.apache.lucene.facet.FacetsCollector;
import org.apache.lucene.facet.FacetsCollectorManager;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.LabelAndValue;
import org.apache.lucene.facet.RandomSamplingFacetsCollector;
import org.apache.lucene.facet.sortedset.ConcurrentSortedSetDocValuesFacetCounts;
import org.apache.lucene.facet.sortedset.DefaultSortedSetDocValuesReaderState;
import org.apache.lucene.facet.sortedset.SortedSetDocValuesFacetCounts;
import org.apache.lucene.facet.sortedset.SortedSetDocValuesFacetField;
import org.apache.lucene.facet.sortedset.SortedSetDocValuesReaderState;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IOUtils;

/**
 * M13 T13.4d2: sorted-set facet counting, drill-sideways over a sorted-set state, random sampling
 * and the {@code Sort} overloads. Writes {@code sortedset_facets/}:
 *
 * <ul>
 *   <li>{@code multi/}: {@link #DOCS} documents in four segments of 900, 600, 200 and 40
 *       (deletions in the first three; the last has no facet field at all) and {@code single/}:
 *       one segment of 300 documents without deletions. Each document ({@link #doc}) carries,
 *       through {@link SortedSetDocValuesFacetField} and {@link #config}: a flat {@code Author}
 *       (missing from some documents), a multi-valued {@code Tag}, a multi-valued {@code Genre}
 *       requiring its dim count, a hierarchical {@code Publish Date} two or three levels deep, a
 *       flat {@code Big} of about 3,000 values (wide enough for a rare query to remap ordinals as
 *       it goes), a flat {@code Color} in its own index field {@code $color}; plus a {@code body}
 *       text field to score, two numeric fields to sort by and a {@code SORTED} field.
 *   <li>{@code state.tsv}: each state's size and dims: a flat dim's {@code OrdRange}, a
 *       hierarchical dim's tree (each ordinal's children, depth first); the refusals.
 *   <li>{@code facets.tsv}: index, counter ({@code ssdv}: {@code SortedSetDocValuesFacetCounts},
 *       {@code conc}: {@code ConcurrentSortedSetDocValuesFacetCounts}), state field, query
 *       ({@code countall} is the no-hits constructor), call and result.
 *   <li>{@code sideways.tsv}: {@code DrillSideways} over the {@code $facets} state: base,
 *       drill-down, mode ({@code seq}, {@code conc}, {@code atonce}), then the hits, the
 *       drill-down and sideways digests, {@code getAllDims(3)} and some {@code getTopChildren(3,
 *       dim)}.
 *   <li>{@code sorted.tsv}: {@code FacetsCollectorManager.search}/{@code searchAfter} and {@code
 *       DrillSideways.search(query, filter, after, topN, sort, doDocScores)} with a {@code Sort}.
 *   <li>{@code sampling.tsv}: {@code RandomSamplingFacetsCollector}: the original and sampled
 *       documents, the sampling rate, sorted-set counts over the sample and their {@code
 *       amortizeFacetCounts}.
 *   <li>{@code errors.tsv}: refusals and their messages.
 * </ul>
 *
 * A result is {@code dim|path|value|childCount|label=value=count,...} with a number {@code
 * i<int>}; a list of results joins them with {@code ;} and leaves {@code null} entries out. A
 * digest is FNV-1a (64-bit) over the global document ids in order; a double or a float is its bits
 * in hex. A string's control characters print as {@code \\u00XX}.
 */
public class GenSortedSetFacets {
  static final int[] SEGMENTS = {900, 600, 200, 40};
  static final int DOCS = 1740;
  static final int NO_FACETS_FROM = 1700;
  static final int SINGLE_DOCS = 300;

  static FacetsConfig config() {
    FacetsConfig c = new FacetsConfig();
    c.setHierarchical("Publish Date", true);
    c.setMultiValued("Tag", true);
    c.setMultiValued("Genre", true);
    c.setRequireDimCount("Genre", true);
    c.setIndexFieldName("Color", "$color");
    return c;
  }

  static String body(int i) {
    StringBuilder sb = new StringBuilder();
    for (int k = 0; k < i % 3 + 1; k++) sb.append("alpha ");
    if (i % 4 == 0) sb.append("beta ");
    if (i % 5 < 2) sb.append("gamma ");
    for (int k = 0; k < i % 7; k++) sb.append("w").append(i % 11).append(' ');
    return sb.toString().trim();
  }

  static Document doc(int i, boolean facets) {
    Document d = new Document();
    d.add(new StringField("id", "d" + i, Field.Store.NO));
    d.add(new StringField("q", "all", Field.Store.NO));
    if (i % 2 == 0) d.add(new StringField("q", "even", Field.Store.NO));
    if (i % 53 == 7) d.add(new StringField("q", "rare", Field.Store.NO));
    if (i % 11 == 4) d.add(new StringField("del", "y", Field.Store.NO));
    d.add(new TextField("body", body(i), Field.Store.NO));
    d.add(new NumericDocValuesField("num", (i * 37L) % 101));
    d.add(new NumericDocValuesField("num2", i % 7));
    d.add(new SortedDocValuesField("sorted", new BytesRef("s" + (i % 5))));
    if (!facets) return d;
    if (i % 17 != 0) d.add(new SortedSetDocValuesFacetField("Author", "a" + (i * 7 % 13)));
    d.add(new SortedSetDocValuesFacetField("Tag", "t" + (i % 5)));
    d.add(new SortedSetDocValuesFacetField("Tag", "t" + (i % 3 + 5)));
    d.add(new SortedSetDocValuesFacetField("Genre", "g" + (i % 4)));
    if (i % 3 == 0) d.add(new SortedSetDocValuesFacetField("Genre", "g" + (i % 6 + 4)));
    String year = "" + (2000 + i % 4);
    String month = "" + (1 + i % 12);
    if (i % 5 == 0) {
      d.add(new SortedSetDocValuesFacetField("Publish Date", year, month));
    } else {
      d.add(new SortedSetDocValuesFacetField("Publish Date", year, month, "" + (1 + i % 28)));
    }
    d.add(new SortedSetDocValuesFacetField("Big", "b" + (i * 7919 % 3000)));
    d.add(new SortedSetDocValuesFacetField("Color", "c" + (i % 3)));
    return d;
  }

  static Map<String, Query> queries() {
    Map<String, Query> m = new LinkedHashMap<>();
    m.put("all", new MatchAllDocsQuery());
    m.put("even", new TermQuery(new Term("q", "even")));
    m.put("rare", new TermQuery(new Term("q", "rare")));
    m.put("none", new TermQuery(new Term("q", "none")));
    return m;
  }

  static Sort sort(String name) {
    switch (name) {
      case "num":
        return new Sort(new SortField("num", SortField.Type.LONG));
      case "num_rev":
        return new Sort(
            new SortField("num", SortField.Type.LONG, true),
            new SortField("num2", SortField.Type.LONG));
      default:
        throw new AssertionError(name);
    }
  }

  // ---------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------

  static String esc(String s) {
    StringBuilder sb = new StringBuilder();
    for (char c : s.toCharArray()) {
      if (c < 0x20 || c == '\\') sb.append(String.format("\\u%04x", (int) c));
      else sb.append(c);
    }
    return sb.toString();
  }

  static String num(Number n) {
    if (n instanceof Float f) return "f" + Integer.toHexString(Float.floatToIntBits(f));
    return "i" + n.intValue();
  }

  static String res(FacetResult r) {
    if (r == null) return "null";
    StringBuilder children = new StringBuilder();
    for (LabelAndValue lv : r.labelValues) {
      if (children.length() > 0) children.append(',');
      children.append(esc(lv.label)).append('=').append(num(lv.value)).append('=').append(lv.count);
    }
    return esc(r.dim)
        + "|"
        + esc(String.join("/", r.path))
        + "|"
        + num(r.value)
        + "|"
        + r.childCount
        + "|"
        + children;
  }

  static String list(List<FacetResult> l) {
    StringBuilder sb = new StringBuilder();
    for (FacetResult r : l) {
      if (r == null) continue; // the Rust port has no null entry
      if (sb.length() > 0) sb.append(';');
      sb.append(res(r));
    }
    return sb.toString();
  }

  static String topDocs(TopDocs t) {
    StringBuilder sb = new StringBuilder();
    sb.append(t.totalHits.value()).append(' ').append(t.totalHits.relation()).append(' ');
    for (int i = 0; i < t.scoreDocs.length; i++) {
      ScoreDoc sd = t.scoreDocs[i];
      if (i > 0) sb.append(',');
      sb.append(sd.doc);
      if (sd instanceof FieldDoc fd) {
        sb.append(':');
        for (int k = 0; k < fd.fields.length; k++) {
          if (k > 0) sb.append('/');
          sb.append(fd.fields[k]);
        }
      }
      sb.append(':').append(Integer.toHexString(Float.floatToIntBits(sd.score)));
    }
    return sb.toString();
  }

  /** Hit count, set-bit count and FNV-1a of the matching docs' global document ids, ascending. */
  static String digest(List<FacetsCollector.MatchingDocs> list) throws Exception {
    List<FacetsCollector.MatchingDocs> mds = new ArrayList<>(list);
    mds.sort(Comparator.comparingInt(m -> m.context().docBase));
    long h = 0xcbf29ce484222325L;
    long hits = 0;
    long n = 0;
    for (FacetsCollector.MatchingDocs m : mds) {
      hits += m.totalHits();
      DocIdSetIterator it = m.bits().iterator();
      for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
        h ^= m.context().docBase + d;
        h *= 0x100000001b3L;
        n++;
      }
    }
    return hits + ":" + n + ":" + Long.toHexString(h);
  }

  interface Call {
    Object run() throws Exception;
  }

  static String attempt(Call c) {
    try {
      Object o = c.run();
      if (o instanceof FacetResult r) return res(r);
      if (o instanceof Number n) return num(n);
      if (o instanceof List<?> l) {
        @SuppressWarnings("unchecked")
        List<FacetResult> rl = (List<FacetResult>) l;
        return list(rl);
      }
      return String.valueOf(o);
    } catch (IllegalArgumentException e) {
      return "IAE:" + esc(e.getMessage());
    } catch (IllegalStateException e) {
      return "ISE:" + esc(e.getMessage());
    } catch (UnsupportedOperationException e) {
      return "UOE:" + esc(e.getMessage());
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  static void line(StringBuilder sb, String... cols) {
    sb.append(String.join("\t", cols)).append('\n');
  }

  static void write(Path p, String s) throws Exception {
    Files.writeString(p, s, StandardCharsets.UTF_8);
  }

  static void writeIndex(Path dir, int[] segments, boolean deletes, int noFacetsFrom)
      throws Exception {
    Files.createDirectories(dir);
    FacetsConfig config = config();
    try (Directory d = FSDirectory.open(dir)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      IndexWriter w = new IndexWriter(d, cfg);
      int i = 0;
      for (int size : segments) {
        for (int k = 0; k < size; k++, i++) w.addDocument(config.build(doc(i, i < noFacetsFrom)));
        w.flush();
      }
      if (deletes) w.deleteDocuments(new Term("del", "y"));
      w.commit();
      w.close();
    }
  }

  // ---------------------------------------------------------------------
  // The state
  // ---------------------------------------------------------------------

  static void tree(StringBuilder sb, SortedSetDocValuesReaderState.DimTree t, int ord) {
    PrimitiveIterator.OfInt it = t.iterator(ord);
    List<Integer> kids = new ArrayList<>();
    while (it.hasNext()) kids.add(it.next());
    if (kids.isEmpty()) return;
    sb.append(' ').append(ord).append(':');
    for (int k = 0; k < kids.size(); k++) {
      if (k > 0) sb.append(',');
      sb.append(kids.get(k));
    }
    for (int kid : kids) tree(sb, t, kid);
  }

  static void state(StringBuilder sb, String index, SortedSetDocValuesReaderState s) {
    List<String> dims = new ArrayList<>();
    for (String d : s.getDims()) dims.add(d);
    dims.sort(Comparator.naturalOrder());
    line(sb, index, s.getField(), "size", "" + s.getSize());
    FacetsConfig config = s.getFacetsConfig();
    for (String dim : dims) {
      if (config.getDimConfig(dim).hierarchical) {
        SortedSetDocValuesReaderState.DimTree t = s.getDimTree(dim);
        StringBuilder tb = new StringBuilder();
        tb.append(t.dimStartOrd);
        tree(tb, t, t.dimStartOrd);
        line(sb, index, s.getField(), "tree", esc(dim), tb.toString());
        line(sb, index, s.getField(), "ordRange", esc(dim), attempt(() -> s.getOrdRange(dim)));
      } else {
        SortedSetDocValuesReaderState.OrdRange r = s.getOrdRange(dim);
        line(sb, index, s.getField(), "range", esc(dim), r.start() + "-" + r.end());
        line(sb, index, s.getField(), "dimTree", esc(dim), attempt(() -> s.getDimTree(dim)));
      }
    }
    line(sb, index, s.getField(), "absentRange", attempt(() -> String.valueOf(s.getOrdRange("Nope"))));
  }

  // ---------------------------------------------------------------------
  // Counting
  // ---------------------------------------------------------------------

  static void calls(StringBuilder sb, String pre, Facets f, boolean colorField) {
    if (colorField) {
      line(sb, pre, "dims5", attempt(() -> f.getAllDims(5)));
      line(sb, pre, "topdims2_3", attempt(() -> f.getTopDims(2, 3)));
      line(sb, pre, "top3:Color", attempt(() -> f.getTopChildren(3, "Color")));
      line(sb, pre, "all:Color", attempt(() -> f.getAllChildren("Color")));
      line(sb, pre, "spec:Color/c1", attempt(() -> f.getSpecificValue("Color", "c1")));
      line(sb, pre, "top3:Author", attempt(() -> f.getTopChildren(3, "Author")));
      return;
    }
    line(sb, pre, "dims5", attempt(() -> f.getAllDims(5)));
    line(sb, pre, "dims1", attempt(() -> f.getAllDims(1)));
    line(sb, pre, "topdims2_3", attempt(() -> f.getTopDims(2, 3)));
    line(sb, pre, "topdims10_2", attempt(() -> f.getTopDims(10, 2)));
    line(sb, pre, "topdims1_1", attempt(() -> f.getTopDims(1, 1)));
    line(sb, pre, "topdims4_5", attempt(() -> f.getTopDims(4, 5)));
    for (String dim : new String[] {"Author", "Tag", "Genre", "Publish Date", "Nope", "Color"}) {
      line(sb, pre, "top3:" + dim, attempt(() -> f.getTopChildren(3, dim)));
      line(sb, pre, "all:" + dim, attempt(() -> f.getAllChildren(dim)));
    }
    line(sb, pre, "top5:Big", attempt(() -> f.getTopChildren(5, "Big")));
    line(sb, pre, "top1:Big", attempt(() -> f.getTopChildren(1, "Big")));
    line(sb, pre, "top1000:Tag", attempt(() -> f.getTopChildren(1000, "Tag")));
    String[][] paths = {
      {"Publish Date", "2001"},
      {"Publish Date", "2001", "3"},
      {"Publish Date", "2003", "12"},
      {"Publish Date", "1999"},
      {"Publish Date", "2001", "3", "3"},
    };
    for (String[] p : paths) {
      String[] rest = Arrays.copyOfRange(p, 1, p.length);
      String name = esc(String.join("/", p));
      line(sb, pre, "top3:" + name, attempt(() -> f.getTopChildren(3, p[0], rest)));
      line(sb, pre, "all:" + name, attempt(() -> f.getAllChildren(p[0], rest)));
    }
    String[][] specs = {
      {"Author", "a3"},
      {"Tag", "t1"},
      {"Tag", "zz"},
      {"Genre", "g1"},
      {"Genre", "g9"},
      {"Genre"},
      {"Publish Date"},
      {"Publish Date", "2001"},
      {"Publish Date", "2001", "3"},
      {"Publish Date", "2001", "3", "15"},
      {"Big", "b7"},
      {"Nope", "x"},
    };
    for (String[] p : specs) {
      String[] rest = Arrays.copyOfRange(p, 1, p.length);
      line(
          sb,
          pre,
          "spec:" + esc(String.join("/", p)),
          attempt(() -> f.getSpecificValue(p[0], rest)));
    }
    line(sb, pre, "err:top0", attempt(() -> f.getTopChildren(0, "Author")));
    line(sb, pre, "err:dims0", attempt(() -> f.getAllDims(0)));
    line(sb, pre, "err:topdims0", attempt(() -> f.getTopDims(0, 1)));
    line(sb, pre, "err:topdims_children0", attempt(() -> f.getTopDims(1, 0)));
    line(sb, pre, "err:flatpath", attempt(() -> f.getTopChildren(3, "Author", "x")));
    line(sb, pre, "err:flatpath_all", attempt(() -> f.getAllChildren("Author", "x")));
    line(sb, pre, "err:spec2", attempt(() -> f.getSpecificValue("Author", "a3", "x")));
  }

  static void counts(
      StringBuilder sb,
      String index,
      DirectoryReader reader,
      SortedSetDocValuesReaderState state,
      ExecutorService exec)
      throws Exception {
    IndexSearcher searcher = new IndexSearcher(reader);
    boolean color = state.getField().equals("$color");
    {
      String pre = index + "\tssdv\t" + state.getField() + "\tcountall";
      calls(sb, pre, new SortedSetDocValuesFacetCounts(state), color);
      pre = index + "\tconc\t" + state.getField() + "\tcountall";
      calls(sb, pre, new ConcurrentSortedSetDocValuesFacetCounts(state, exec), color);
    }
    for (Map.Entry<String, Query> q : queries().entrySet()) {
      FacetsCollector fc = searcher.search(q.getValue(), new FacetsCollectorManager());
      String pre = index + "\tssdv\t" + state.getField() + "\t" + q.getKey();
      calls(sb, pre, new SortedSetDocValuesFacetCounts(state, fc), color);
      pre = index + "\tconc\t" + state.getField() + "\t" + q.getKey();
      calls(sb, pre, new ConcurrentSortedSetDocValuesFacetCounts(state, fc, exec), color);
    }
  }

  // ---------------------------------------------------------------------
  // Drill sideways
  // ---------------------------------------------------------------------

  static Map<String, Query> bases() {
    Map<String, Query> m = new LinkedHashMap<>();
    m.put("null", null);
    m.put("all", new MatchAllDocsQuery());
    m.put("even", new TermQuery(new Term("q", "even")));
    m.put("rare", new TermQuery(new Term("q", "rare")));
    m.put("alpha", new TermQuery(new Term("body", "alpha")));
    return m;
  }

  static Map<String, String[][]> drillDowns() {
    Map<String, String[][]> m = new LinkedHashMap<>();
    m.put("none", new String[][] {});
    m.put("author", new String[][] {{"Author", "a3"}});
    m.put("author_or", new String[][] {{"Author", "a3"}, {"Author", "a5"}});
    m.put("author_tag", new String[][] {{"Author", "a3"}, {"Tag", "t1"}});
    m.put("date", new String[][] {{"Publish Date", "2001"}});
    m.put("deep", new String[][] {{"Publish Date", "2001", "3"}});
    m.put("genre_tag", new String[][] {{"Genre", "g1"}, {"Tag", "t6"}});
    m.put("absent", new String[][] {{"Author", "zz"}});
    return m;
  }

  static DrillDownQuery ddq(FacetsConfig config, Query base, String[][] steps) {
    DrillDownQuery q = base == null ? new DrillDownQuery(config) : new DrillDownQuery(config, base);
    for (String[] step : steps) q.add(step[0], Arrays.copyOfRange(step, 1, step.length));
    return q;
  }

  static String sideways(DrillSideways.DrillSidewaysResult r) throws Exception {
    if (r.drillSidewaysDims == null) return "-";
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < r.drillSidewaysDims.length; i++) {
      if (i > 0) sb.append(';');
      sb.append(r.drillSidewaysDims[i])
          .append('=')
          .append(digest(r.drillSidewaysFacetsCollector[i].getMatchingDocs()));
    }
    return sb.toString();
  }

  static String facets(Facets f, String[][] steps) throws Exception {
    List<String> dims = new ArrayList<>();
    for (String[] s : steps) if (!dims.contains(s[0])) dims.add(s[0]);
    for (String d : new String[] {"Author", "Tag", "Genre"}) if (!dims.contains(d)) dims.add(d);
    StringBuilder top = new StringBuilder();
    for (String d : dims) {
      if (top.length() > 0) top.append(';');
      top.append(attempt(() -> f.getTopChildren(3, d)));
    }
    return attempt(() -> f.getAllDims(3)) + "\t" + top;
  }

  static String dsResult(DrillSideways.DrillSidewaysResult r, String[][] steps) throws Exception {
    return topDocs(r.hits)
        + "\t"
        + digest(r.drillDownFacetsCollector.getMatchingDocs())
        + "\t"
        + sideways(r)
        + "\t"
        + facets(r.facets, steps);
  }

  // ---------------------------------------------------------------------
  // Sampling
  // ---------------------------------------------------------------------

  static String sampled(
      IndexSearcher searcher,
      FacetsConfig config,
      SortedSetDocValuesReaderState state,
      RandomSamplingFacetsCollector c)
      throws Exception {
    StringBuilder sb = new StringBuilder();
    sb.append(digest(c.getOriginalMatchingDocs())).append('\t');
    sb.append(digest(c.getMatchingDocs())).append('\t');
    sb.append(Long.toHexString(Double.doubleToLongBits(c.getSamplingRate())));
    Facets f = new SortedSetDocValuesFacetCounts(state, c);
    String[][] paths = {{"Author"}, {"Tag"}, {"Genre"}, {"Publish Date"}, {"Publish Date", "2001"}};
    for (String[] p : paths) {
      String[] rest = Arrays.copyOfRange(p, 1, p.length);
      FacetResult r = f.getTopChildren(5, p[0], rest);
      sb.append('\t').append(res(r));
      sb.append('\t').append(res(c.amortizeFacetCounts(r, config, searcher)));
    }
    return sb.toString();
  }

  // ---------------------------------------------------------------------

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("sortedset_facets");
    if (Files.exists(root)) IOUtils.rm(root);
    writeIndex(root.resolve("multi"), SEGMENTS, true, NO_FACETS_FROM);
    writeIndex(root.resolve("single"), new int[] {SINGLE_DOCS}, false, SINGLE_DOCS);

    StringBuilder stateTsv = new StringBuilder();
    StringBuilder facetsTsv = new StringBuilder();
    StringBuilder sidewaysTsv = new StringBuilder();
    StringBuilder sortedTsv = new StringBuilder();
    StringBuilder samplingTsv = new StringBuilder();
    StringBuilder errors = new StringBuilder();
    FacetsConfig config = config();
    ExecutorService exec = Executors.newFixedThreadPool(3);
    try (Directory md = FSDirectory.open(root.resolve("multi"));
        Directory sd = FSDirectory.open(root.resolve("single"));
        DirectoryReader multi = DirectoryReader.open(md);
        DirectoryReader single = DirectoryReader.open(sd)) {
      if (multi.leaves().size() != SEGMENTS.length) throw new AssertionError("segments");
      Map<String, DirectoryReader> indexes = new LinkedHashMap<>();
      indexes.put("multi", multi);
      indexes.put("single", single);
      Map<String, SortedSetDocValuesReaderState> states = new LinkedHashMap<>();
      for (Map.Entry<String, DirectoryReader> e : indexes.entrySet()) {
        for (String field : new String[] {"$facets", "$color"}) {
          SortedSetDocValuesReaderState s =
              new DefaultSortedSetDocValuesReaderState(e.getValue(), field, config);
          states.put(e.getKey() + field, s);
          state(stateTsv, e.getKey(), s);
          counts(facetsTsv, e.getKey(), e.getValue(), s, exec);
        }
      }

      // Refusals.
      line(
          errors,
          "state:absent",
          attempt(() -> new DefaultSortedSetDocValuesReaderState(multi, "$nope", config)));
      line(
          errors,
          "state:sorted",
          attempt(() -> new DefaultSortedSetDocValuesReaderState(multi, "sorted", config)));
      line(
          errors,
          "state:sorted_single",
          attempt(() -> new DefaultSortedSetDocValuesReaderState(single, "sorted", config)));
      line(
          errors,
          "state:id",
          attempt(() -> new DefaultSortedSetDocValuesReaderState(multi, "id", config)));
      {
        FacetsCollector other =
            new IndexSearcher(single).search(new MatchAllDocsQuery(), new FacetsCollectorManager());
        SortedSetDocValuesReaderState ms = states.get("multi$facets");
        line(
            errors,
            "count:mismatch",
            attempt(() -> new SortedSetDocValuesFacetCounts(ms, other)));
        line(
            errors,
            "conc:mismatch",
            attempt(() -> new ConcurrentSortedSetDocValuesFacetCounts(ms, other, exec)));
      }

      IndexSearcher searcher = new IndexSearcher(multi);
      if (searcher.getSlices().length != 1) throw new AssertionError("one slice expected");
      SortedSetDocValuesReaderState state = states.get("multi$facets");

      // Drill sideways.
      DrillSideways seq = new DrillSideways(searcher, config, state);
      DrillSideways conc = new DrillSideways(searcher, config, null, state, exec);
      DrillSideways atOnce =
          new DrillSideways(searcher, config, state) {
            @Override
            protected boolean scoreSubDocsAtOnce() {
              return true;
            }
          };
      Map<String, DrillSideways> modes = new LinkedHashMap<>();
      modes.put("seq", seq);
      modes.put("conc", conc);
      modes.put("atonce", atOnce);
      for (Map.Entry<String, Query> b : bases().entrySet()) {
        for (Map.Entry<String, String[][]> dd : drillDowns().entrySet()) {
          for (Map.Entry<String, DrillSideways> m : modes.entrySet()) {
            DrillSideways.DrillSidewaysResult r =
                m.getValue().search(ddq(config, b.getValue(), dd.getValue()), 10);
            line(
                sidewaysTsv,
                b.getKey(),
                dd.getKey(),
                m.getKey(),
                dsResult(r, dd.getValue()));
          }
        }
      }

      // Sorting: FacetsCollectorManager.
      for (String qn : new String[] {"all", "even", "rare"}) {
        Query q = qn.equals("all") ? new TermQuery(new Term("body", "alpha")) : queries().get(qn);
        for (String sn : new String[] {"num", "num_rev"}) {
          for (boolean scores : new boolean[] {false, true}) {
            String pre = "fcm\t" + qn + "\t" + sn + "\t" + scores;
            FacetsCollectorManager.FacetsResult r =
                FacetsCollectorManager.search(
                    searcher, q, 7, sort(sn), scores, new FacetsCollectorManager());
            line(
                sortedTsv,
                pre,
                "first",
                topDocs(r.topDocs()),
                digest(r.facetsCollector().getMatchingDocs()));
            if (r.topDocs().scoreDocs.length > 3) {
              ScoreDoc after = r.topDocs().scoreDocs[3];
              FacetsCollectorManager.FacetsResult next =
                  FacetsCollectorManager.searchAfter(
                      searcher, after, q, 5, sort(sn), scores, new FacetsCollectorManager());
              line(
                  sortedTsv,
                  pre,
                  "after",
                  topDocs(next.topDocs()),
                  digest(next.facetsCollector().getMatchingDocs()));
            }
          }
        }
        FacetsCollectorManager.FacetsResult zero =
            FacetsCollectorManager.search(searcher, q, 0, sort("num"), new FacetsCollectorManager());
        line(
            sortedTsv,
            "fcm\t" + qn + "\tnum\tfalse",
            "zero",
            topDocs(zero.topDocs()),
            digest(zero.facetsCollector().getMatchingDocs()));
      }
      // Sorting: DrillSideways.
      for (String bn : new String[] {"null", "alpha"}) {
        for (String dn : new String[] {"none", "author", "author_tag"}) {
          String[][] steps = drillDowns().get(dn);
          for (String sn : new String[] {"num", "num_rev"}) {
            for (boolean scores : new boolean[] {false, true}) {
              String pre = "ds\t" + bn + "/" + dn + "\t" + sn + "\t" + scores;
              DrillSideways.DrillSidewaysResult r =
                  seq.search(ddq(config, bases().get(bn), steps), null, null, 6, sort(sn), scores);
              line(sortedTsv, pre, "first", dsResult(r, steps));
              FieldDoc after = (FieldDoc) r.hits.scoreDocs[2];
              DrillSideways.DrillSidewaysResult next =
                  seq.search(
                      ddq(config, bases().get(bn), steps),
                      new TermQuery(new Term("q", "even")),
                      after,
                      4,
                      sort(sn),
                      scores);
              line(sortedTsv, pre, "after_filtered", dsResult(next, steps));
            }
          }
        }
      }
      line(
          errors,
          "fcm:nullsort",
          attempt(
              () ->
                  FacetsCollectorManager.search(
                      searcher, new MatchAllDocsQuery(), 5, null, new FacetsCollectorManager())));
      line(
          errors,
          "fcm:after_past_limit",
          attempt(
              () ->
                  FacetsCollectorManager.searchAfter(
                      searcher,
                      new FieldDoc(DOCS, Float.NaN, new Object[] {0L}),
                      new MatchAllDocsQuery(),
                      5,
                      sort("num"),
                      new FacetsCollectorManager())));

      // Sampling.
      for (String qn : new String[] {"all", "even", "rare"}) {
        Query q = queries().get(qn);
        for (int size : new int[] {0, 1, 10, 50, 200, 5000}) {
          for (long seed : new long[] {0L, 42L, -7L}) {
            RandomSamplingFacetsCollector c =
                searcher.search(q, RandomSamplingFacetsCollector.createManager(size, seed));
            line(samplingTsv, qn, "" + size, "" + seed, sampled(searcher, config, state, c));
          }
        }
      }
    } finally {
      exec.shutdown();
    }

    write(root.resolve("state.tsv"), stateTsv.toString());
    write(root.resolve("facets.tsv"), facetsTsv.toString());
    write(root.resolve("sideways.tsv"), sidewaysTsv.toString());
    write(root.resolve("sorted.tsv"), sortedTsv.toString());
    write(root.resolve("sampling.tsv"), samplingTsv.toString());
    write(root.resolve("errors.tsv"), errors.toString());
  }
}
