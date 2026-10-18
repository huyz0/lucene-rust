import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleDocValuesField;
import org.apache.lucene.document.DoubleRangeDocValuesField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatDocValuesField;
import org.apache.lucene.document.LongRangeDocValuesField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.RangeFieldQuery;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.Facets;
import org.apache.lucene.facet.FacetsCollector;
import org.apache.lucene.facet.FacetsCollectorManager;
import org.apache.lucene.facet.LabelAndValue;
import org.apache.lucene.facet.LongValueFacetCounts;
import org.apache.lucene.facet.MultiDoubleValuesSource;
import org.apache.lucene.facet.MultiLongValuesSource;
import org.apache.lucene.facet.StringDocValuesReaderState;
import org.apache.lucene.facet.StringValueFacetCounts;
import org.apache.lucene.facet.facetset.DimRange;
import org.apache.lucene.facet.facetset.DoubleFacetSet;
import org.apache.lucene.facet.facetset.ExactFacetSetMatcher;
import org.apache.lucene.facet.facetset.FacetSetDecoder;
import org.apache.lucene.facet.facetset.FacetSetMatcher;
import org.apache.lucene.facet.facetset.FacetSetsField;
import org.apache.lucene.facet.facetset.FloatFacetSet;
import org.apache.lucene.facet.facetset.IntFacetSet;
import org.apache.lucene.facet.facetset.LongFacetSet;
import org.apache.lucene.facet.facetset.MatchingFacetSetsCounts;
import org.apache.lucene.facet.facetset.RangeFacetSetMatcher;
import org.apache.lucene.facet.range.DoubleRange;
import org.apache.lucene.facet.range.DoubleRangeFacetCounts;
import org.apache.lucene.facet.range.DynamicRangeUtil;
import org.apache.lucene.facet.range.LongRange;
import org.apache.lucene.facet.range.LongRangeFacetCounts;
import org.apache.lucene.facet.rangeonrange.DoubleRangeOnRangeFacetCounts;
import org.apache.lucene.facet.rangeonrange.LongRangeOnRangeFacetCounts;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IOUtils;

/**
 * M13 T13.4d1: value, range, range-on-range and facet-set counting. Writes {@code value_facets/}:
 *
 * <ul>
 *   <li>{@code multi/}: {@link #DOCS} documents in four segments of 700, 500, 300 and 60
 *       (deletions in each; the last has no facet field at all) and {@code single/}: one segment of
 *       250 documents without deletions (no ordinal map, the no-hits-no-live-docs count). Each
 *       document ({@link #doc}) carries single- and multi-valued numeric, double, float and string
 *       doc values (some values {@code >= 1024}, negative, {@code Long.MIN/MAX_VALUE}, {@code NaN},
 *       infinities, {@code -0.0}; a string field of over 1,024 values), 1- and 2-dimensional long
 *       and 1-dimensional double range doc values, four facet-set fields and three weight fields.
 *   <li>{@code facets.tsv}: index, query, call, result. A call names the counter and its input
 *       then what was asked: {@code top<n>} ({@code getTopChildren}), {@code all}
 *       ({@code getAllChildren}), {@code sorted} ({@code getAllChildrenSortByValue}), {@code
 *       dims<n>} ({@code getAllDims}), {@code str} ({@code toString}), {@code spec:<label>}
 *       ({@code getSpecificValue}). The hash-backed counters iterate their hash tables in an order
 *       Java seeds per process, so for {@code LongValueFacetCounts}'s and {@code
 *       StringValueFacetCounts}'s {@code all} and {@code str} the children / lines are written
 *       sorted (by label / by line); the reader sorts its own the same way.
 *   <li>{@code queries.tsv}: {@code Range.getQuery}: index, name, {@code toString}, hit count and
 *       digest.
 *   <li>{@code dynamic.tsv}: {@code DynamicRangeUtil} over the indexes and over hand-built arrays.
 *   <li>{@code errors.tsv}: refusals: the exception's kind ({@code IAE}, {@code ISE}, {@code
 *       UOE}) and message.
 * </ul>
 *
 * A result is {@code dim|path|value|childCount|label=value=count,...} with a number {@code
 * i<int>}; a digest is FNV-1a (64-bit) over the global document ids in order; a double is its
 * bits in hex. A string's control characters print as {@code \\u00XX}.
 */
public class GenValueFacets {
  static final int[] SEGMENTS = {700, 500, 300, 60};
  static final int DOCS = 1560;
  static final int NO_FACETS_FROM = 1500;
  static final int SINGLE_DOCS = 250;

  static long long1(int i) {
    switch (i % 10) {
      case 0:
        return i % 13;
      case 1:
        return 1024 + i % 5;
      case 2:
        return -(i % 4) - 1;
      case 3:
        return Long.MAX_VALUE;
      case 4:
        return Long.MIN_VALUE;
      case 5:
        return 5000 + i % 3;
      case 6:
        return 1023;
      default:
        return i % 20;
    }
  }

  static double dbl(int i) {
    switch (i % 50) {
      case 1:
        return -0.0;
      case 2:
        return Double.NaN;
      case 3:
        return Double.POSITIVE_INFINITY;
      case 4:
        return Double.NEGATIVE_INFINITY;
      default:
        return (i % 17) * 0.5 - 3.0;
    }
  }

  static String str(int i) {
    if (i % 23 == 22) return "😀x";
    if (i % 23 == 21) return "ﬁx";
    if (i % 23 == 20) return "été";
    return "s" + (i % 23);
  }

  static Document doc(int i) {
    Document d = new Document();
    d.add(new StringField("q", "all", Field.Store.NO));
    if (i % 2 == 0) d.add(new StringField("q", "even", Field.Store.NO));
    if (i % 37 == 5) d.add(new StringField("q", "rare", Field.Store.NO));
    if (i % 11 == 3) d.add(new StringField("del", "y", Field.Store.NO));
    if (i >= NO_FACETS_FROM) return d;
    if (i % 9 != 0) d.add(new NumericDocValuesField("long1", long1(i)));
    long[] lm = {i % 6, i % 6, 2000 + i % 3, -7};
    for (int k = 0; k < i % 4; k++) d.add(new SortedNumericDocValuesField("longm", lm[k]));
    if (i % 17 == 0) d.add(new SortedNumericDocValuesField("longm", Long.MAX_VALUE));
    if (i % 8 != 0) d.add(new DoubleDocValuesField("dbl", dbl(i)));
    double[] dm = {(i % 5) - 2.0, (i % 7) * 1.5};
    for (int k = 0; k < i % 3; k++) {
      d.add(new SortedNumericDocValuesField("dblm", Double.doubleToRawLongBits(dm[k])));
    }
    if (i % 6 != 0) d.add(new FloatDocValuesField("flt", (i % 11) * 0.25f - 1f));
    if (i % 5 != 0) d.add(new SortedDocValuesField("str", new BytesRef(str(i))));
    for (int k = 0; k < i % 3; k++) {
      String v = k == 0 ? "m" + (i % 4) : "m" + (i % 7 + 2);
      d.add(new SortedSetDocValuesField("strm", new BytesRef(v)));
    }
    d.add(new SortedDocValuesField("strhi", new BytesRef(String.format("h%04d", i % 1300))));
    d.add(new SortedSetDocValuesField("strhim", new BytesRef(String.format("x%04d", i % 1100))));
    if (i % 2 == 0) d.add(new SortedSetDocValuesField("strhim", new BytesRef("y" + (i % 40))));
    if (i % 10 != 9) {
      long min = i % 20 - 5;
      d.add(new LongRangeDocValuesField("rorlong", new long[] {min}, new long[] {min + i % 7}));
    }
    if (i % 4 != 1) {
      d.add(
          new LongRangeDocValuesField(
              "rorlong2",
              new long[] {i % 10, i % 5},
              new long[] {i % 10 + 3, i % 5 + i % 4}));
    }
    if (i % 6 != 5) {
      double min = (i % 13) * 0.5 - 2;
      d.add(
          new DoubleRangeDocValuesField(
              "rordbl", new double[] {min}, new double[] {min + (i % 3) * 1.25}));
    }
    if (i % 7 != 6) {
      if (i % 2 == 0) {
        d.add(
            FacetSetsField.create(
                "fs_long",
                new LongFacetSet(i % 5, i % 3),
                new LongFacetSet(i % 4, 7),
                new LongFacetSet(1, 2)));
      } else {
        d.add(
            FacetSetsField.create(
                "fs_long", new LongFacetSet(i % 5, i % 3), new LongFacetSet(i % 4, 7)));
      }
    }
    if (i % 5 != 4) {
      if (i % 3 == 0) {
        d.add(
            FacetSetsField.create(
                "fs_int", new IntFacetSet(i % 3, i % 4, -(i % 2)), new IntFacetSet(1, 2, 0)));
      } else {
        d.add(FacetSetsField.create("fs_int", new IntFacetSet(i % 3, i % 4, -(i % 2))));
      }
    }
    if (i % 4 != 3) {
      if (i % 10 == 0) {
        d.add(
            FacetSetsField.create(
                "fs_dbl", new DoubleFacetSet((i % 9) * 0.5 - 1.0), new DoubleFacetSet(-0.0)));
      } else {
        d.add(FacetSetsField.create("fs_dbl", new DoubleFacetSet((i % 9) * 0.5 - 1.0)));
      }
    }
    if (i % 3 != 2) {
      d.add(
          FacetSetsField.create("fs_flt", new FloatFacetSet((i % 4) * 0.5f, -(i % 3) * 1.5f)));
    }
    if (i % 7 != 2) d.add(new NumericDocValuesField("w", i % 31 == 0 ? -3 : i % 6));
    if (i == 0 || i == 700) d.add(new NumericDocValuesField("wx", Long.MAX_VALUE - 10));
    if (i == 1 || i == 2) d.add(new NumericDocValuesField("wy", Long.MAX_VALUE / 2 + 1));
    return d;
  }

  static Map<String, Query> queries() {
    Map<String, Query> m = new LinkedHashMap<>();
    m.put("all", new MatchAllDocsQuery());
    m.put("even", new TermQuery(new Term("q", "even")));
    m.put("rare", new TermQuery(new Term("q", "rare")));
    m.put("none", new TermQuery(new Term("q", "zzz")));
    return m;
  }

  static final Query FAST = new TermQuery(new Term("q", "even"));

  static LongRange[] longRanges(String name) {
    switch (name) {
      case "excl":
        return new LongRange[] {
          new LongRange("lt0", Long.MIN_VALUE, true, 0, false),
          new LongRange("0-10", 0, true, 10, true),
          new LongRange("10-20", 10, false, 20, true),
          new LongRange("big", 1000, true, Long.MAX_VALUE, true),
        };
      case "gaps":
        return new LongRange[] {
          new LongRange("b", 5, true, 6, true), new LongRange("a", -3, true, -1, true),
        };
      case "overl":
        return new LongRange[] {
          new LongRange("a", 0, true, 5, true),
          new LongRange("b", 3, true, 10, true),
          new LongRange("c", -5, true, 100, false),
          new LongRange("every", Long.MIN_VALUE, true, Long.MAX_VALUE, true),
          new LongRange("pt", 7, true, 7, true),
          new LongRange("hi", 1024, true, 1100, true),
          new LongRange("max", Long.MAX_VALUE, true, Long.MAX_VALUE, true),
        };
      case "empty":
        return new LongRange[0];
      default:
        throw new AssertionError(name);
    }
  }

  static DoubleRange[] doubleRanges(String name) {
    switch (name) {
      case "excl":
        return new DoubleRange[] {
          new DoubleRange("neg", Double.NEGATIVE_INFINITY, true, 0, false),
          new DoubleRange("0-1", 0, true, 1, true),
          new DoubleRange("open", 1, false, Double.POSITIVE_INFINITY, true),
        };
      case "overl":
        return new DoubleRange[] {
          new DoubleRange("a", -1.5, true, 2.0, false),
          new DoubleRange("b", 0.5, true, 3.0, true),
          new DoubleRange("zero", -0.0, true, 0.0, true),
          new DoubleRange("inf", 2.0, true, Double.POSITIVE_INFINITY, true),
          new DoubleRange("a", -1.5, true, 2.0, false),
        };
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

  static String res(FacetResult r, boolean sortChildren) {
    if (r == null) return "null";
    List<String> kids = new ArrayList<>();
    for (LabelAndValue lv : r.labelValues) {
      kids.add(esc(lv.label) + "=" + num(lv.value) + "=" + lv.count);
    }
    if (sortChildren) kids.sort(Comparator.naturalOrder());
    return esc(r.dim)
        + "|"
        + String.join("/", r.path)
        + "|"
        + num(r.value)
        + "|"
        + r.childCount
        + "|"
        + String.join(",", kids);
  }

  static String list(List<FacetResult> l) {
    StringBuilder sb = new StringBuilder();
    for (FacetResult r : l) {
      if (sb.length() > 0) sb.append(';');
      sb.append(res(r, false));
    }
    return sb.toString();
  }

  /** {@code toString} with its lines after the first sorted (hash order is per process). */
  static String sortedLines(String s) {
    String[] lines = s.split("\n", -1);
    List<String> body = new ArrayList<>(Arrays.asList(lines).subList(1, lines.length));
    body.sort(Comparator.naturalOrder());
    return esc(lines[0] + "\n" + String.join("\n", body));
  }

  static String digest(FacetsCollector fc) throws Exception {
    List<FacetsCollector.MatchingDocs> mds = new ArrayList<>(fc.getMatchingDocs());
    mds.sort(Comparator.comparingInt(m -> m.context().docBase));
    long h = 0xcbf29ce484222325L;
    long hits = 0;
    for (FacetsCollector.MatchingDocs m : mds) {
      hits += m.totalHits();
      DocIdSetIterator it = m.bits().iterator();
      for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
        h ^= m.context().docBase + d;
        h *= 0x100000001b3L;
      }
    }
    return hits + ":" + Long.toHexString(h);
  }

  interface Call {
    Object run() throws Exception;
  }

  static String attempt(Call c) {
    try {
      Object o = c.run();
      if (o instanceof FacetResult r) return res(r, false);
      if (o instanceof Number n) return num(n);
      return String.valueOf(o);
    } catch (IllegalArgumentException e) {
      return "IAE:" + esc(e.getMessage());
    } catch (IllegalStateException e) {
      return "ISE:" + esc(e.getMessage());
    } catch (UnsupportedOperationException e) {
      return "UOE";
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  static void line(StringBuilder sb, String index, String query, String call, String result) {
    sb.append(index)
        .append('\t')
        .append(query)
        .append('\t')
        .append(call)
        .append('\t')
        .append(result)
        .append('\n');
  }

  /** The calls every single-dim counter answers, its dim {@code dim}. */
  static void standard(
      StringBuilder sb, String index, String query, String name, Facets f, String dim) {
    line(sb, index, query, name + " top1", attempt(() -> f.getTopChildren(1, dim)));
    line(sb, index, query, name + " top3", attempt(() -> f.getTopChildren(3, dim)));
    line(sb, index, query, name + " top100", attempt(() -> f.getTopChildren(100, dim)));
    line(sb, index, query, name + " dims2", attempt(() -> list(f.getAllDims(2))));
    line(sb, index, query, name + " top0", attempt(() -> f.getTopChildren(0, dim)));
    line(sb, index, query, name + " baddim", attempt(() -> f.getTopChildren(3, "nope")));
    line(sb, index, query, name + " badpath", attempt(() -> f.getTopChildren(3, dim, "x")));
    line(sb, index, query, name + " allbaddim", attempt(() -> f.getAllChildren("nope")));
    line(sb, index, query, name + " allbadpath", attempt(() -> f.getAllChildren(dim, "x")));
    line(sb, index, query, name + " spec", attempt(() -> f.getSpecificValue(dim, "1")));
  }

  static void longValues(StringBuilder sb, String index, String query, String name, Call make) {
    LongValueFacetCounts f;
    try {
      f = (LongValueFacetCounts) make.run();
    } catch (Exception e) {
      throw new AssertionError(name, e);
    }
    String dim = name.substring(name.indexOf(':') + 1, name.indexOf('@'));
    standard(sb, index, query, name, f, dim);
    line(sb, index, query, name + " all", attempt(() -> res(f.getAllChildren(dim), true)));
    line(sb, index, query, name + " sorted", attempt(() -> res(f.getAllChildrenSortByValue(), false)));
    line(sb, index, query, name + " str", sortedLines(f.toString()));
  }

  static void ranges(StringBuilder sb, String index, String query, String name, Facets f, String dim) {
    standard(sb, index, query, name, f, dim);
    line(sb, index, query, name + " all", attempt(() -> f.getAllChildren(dim)));
    line(sb, index, query, name + " str", esc(f.toString()));
  }

  static FacetSetMatcher[] matchers(String field) {
    switch (field) {
      case "fs_long":
        return new FacetSetMatcher[] {
          new ExactFacetSetMatcher("e12", new LongFacetSet(1, 2)),
          new RangeFacetSetMatcher(
              "r", DimRange.fromLongs(0, true, 2, true), DimRange.fromLongs(0, true, 10, false)),
          new ExactFacetSetMatcher("e37", new LongFacetSet(3, 7)),
          new RangeFacetSetMatcher(
              "none",
              DimRange.fromLongs(100, true, 200, true),
              DimRange.fromLongs(0, false, 1, true)),
          new ExactFacetSetMatcher("e00", new LongFacetSet(0, 0)),
        };
      case "fs_int":
        return new FacetSetMatcher[] {
          new ExactFacetSetMatcher("i120", new IntFacetSet(1, 2, 0)),
          new RangeFacetSetMatcher(
              "neg",
              DimRange.fromLongs(0, true, 2, true),
              DimRange.fromLongs(Long.MIN_VALUE, true, Long.MAX_VALUE, true),
              DimRange.fromLongs(-1, true, 0, false)),
          new ExactFacetSetMatcher("i000", new IntFacetSet(0, 0, 0)),
        };
      case "fs_dbl":
        return new FacetSetMatcher[] {
          new ExactFacetSetMatcher("half", new DoubleFacetSet(0.5)),
          new RangeFacetSetMatcher("neg", DimRange.fromDoubles(-1.0, true, 0.0, false)),
          new RangeFacetSetMatcher("pos", DimRange.fromDoubles(0.0, true, 10.0, true)),
          new ExactFacetSetMatcher("negzero", new DoubleFacetSet(-0.0)),
        };
      case "fs_flt":
        return new FacetSetMatcher[] {
          new ExactFacetSetMatcher("f", new FloatFacetSet(0.5f, -1.5f)),
          new RangeFacetSetMatcher(
              "rf",
              DimRange.fromFloats(0f, false, 1.0f, true),
              DimRange.fromFloats(Float.NEGATIVE_INFINITY, true, 0f, false)),
        };
      default:
        throw new AssertionError(field);
    }
  }

  static FacetSetDecoder decoder(String field) {
    return field.equals("fs_int") || field.equals("fs_flt")
        ? FacetSetDecoder::decodeInts
        : FacetSetDecoder::decodeLongs;
  }

  static void writeIndex(Path dir, int[] segments, boolean deletes) throws Exception {
    Files.createDirectories(dir);
    try (Directory d = FSDirectory.open(dir)) {
      IndexWriterConfig cfg = new IndexWriterConfig();
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      IndexWriter w = new IndexWriter(d, cfg);
      int i = 0;
      for (int size : segments) {
        for (int k = 0; k < size; k++, i++) w.addDocument(doc(i));
        w.flush();
      }
      if (deletes) w.deleteDocuments(new Term("del", "y"));
      w.commit();
      w.close();
    }
  }

  static String ranges(List<DynamicRangeUtil.DynamicRangeInfo> l) {
    StringBuilder sb = new StringBuilder();
    for (DynamicRangeUtil.DynamicRangeInfo r : l) {
      if (sb.length() > 0) sb.append(';');
      sb.append(r.count())
          .append(',')
          .append(r.weight())
          .append(',')
          .append(r.min())
          .append(',')
          .append(r.max())
          .append(',')
          .append(Long.toHexString(Double.doubleToLongBits(r.centroid())));
    }
    return sb.toString();
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("value_facets");
    if (Files.exists(root)) IOUtils.rm(root);
    writeIndex(root.resolve("multi"), SEGMENTS, true);
    writeIndex(root.resolve("single"), new int[] {SINGLE_DOCS}, false);

    StringBuilder facets = new StringBuilder();
    StringBuilder queries = new StringBuilder();
    StringBuilder dynamic = new StringBuilder();
    StringBuilder errors = new StringBuilder();
    ExecutorService exec = Executors.newFixedThreadPool(2);
    try (Directory md = FSDirectory.open(root.resolve("multi"));
        Directory sd = FSDirectory.open(root.resolve("single"));
        DirectoryReader multi = DirectoryReader.open(md);
        DirectoryReader single = DirectoryReader.open(sd)) {
      if (multi.leaves().size() != SEGMENTS.length) throw new AssertionError("segments");
      Map<String, DirectoryReader> readers = new LinkedHashMap<>();
      readers.put("multi", multi);
      readers.put("single", single);
      for (Map.Entry<String, DirectoryReader> re : readers.entrySet()) {
        String index = re.getKey();
        DirectoryReader reader = re.getValue();
        IndexSearcher searcher = new IndexSearcher(reader);

        // Count-all forms.
        longValues(facets, index, "-", "lv:long1@field", () -> new LongValueFacetCounts("long1", reader));
        longValues(facets, index, "-", "lv:longm@field", () -> new LongValueFacetCounts("longm", reader));
        longValues(
            facets, index, "-", "lv:long1@src",
            () -> new LongValueFacetCounts("long1", LongValuesSource.fromLongField("long1"), reader));
        longValues(
            facets, index, "-", "lv:longm@multi",
            () -> new LongValueFacetCounts("longm", MultiLongValuesSource.fromLongField("longm"), reader));
        longValues(
            facets, index, "-", "lv:long1@single",
            () ->
                new LongValueFacetCounts(
                    "long1",
                    MultiLongValuesSource.fromSingleValued(LongValuesSource.fromLongField("long1")),
                    reader));
        longValues(facets, index, "-", "lv:nofield@field", () -> new LongValueFacetCounts("nofield", reader));
        for (String field : new String[] {"str", "strm", "strhi", "strhim", "nofield"}) {
          StringDocValuesReaderState state = new StringDocValuesReaderState(reader, field);
          StringValueFacetCounts f = new StringValueFacetCounts(state);
          stringValues(facets, index, "-", "sv:" + field, f, field);
        }

        for (Map.Entry<String, Query> qe : queries().entrySet()) {
          String query = qe.getKey();
          FacetsCollector fc = searcher.search(qe.getValue(), new FacetsCollectorManager());
          FacetsCollector fcAll = searcher.search(new MatchAllDocsQuery(), new FacetsCollectorManager());
          line(facets, index, query, "hits", digest(fc));

          // LongValueFacetCounts.
          longValues(facets, index, query, "lv:long1@field", () -> new LongValueFacetCounts("long1", fc));
          longValues(facets, index, query, "lv:longm@field", () -> new LongValueFacetCounts("longm", fc));
          longValues(facets, index, query, "lv:nofield@field", () -> new LongValueFacetCounts("nofield", fc));
          longValues(
              facets, index, query, "lv:long1@src",
              () -> new LongValueFacetCounts("long1", LongValuesSource.fromLongField("long1"), fc));
          longValues(
              facets, index, query, "lv:longm@multi",
              () -> new LongValueFacetCounts("longm", MultiLongValuesSource.fromLongField("longm"), fc));
          longValues(
              facets, index, query, "lv:long1@single",
              () ->
                  new LongValueFacetCounts(
                      "long1",
                      MultiLongValuesSource.fromSingleValued(LongValuesSource.fromLongField("long1")),
                      fc));
          longValues(
              facets, index, query, "lv:dblm@tolong",
              () ->
                  new LongValueFacetCounts(
                      "dblm", MultiDoubleValuesSource.fromDoubleField("dblm").toMultiLongValuesSource(), fc));
          longValues(
              facets, index, query, "lv:flt@tolong",
              () ->
                  new LongValueFacetCounts(
                      "flt", MultiDoubleValuesSource.fromFloatField("flt").toMultiLongValuesSource(), fc));
          longValues(
              facets, index, query, "lv:dblm@sortable",
              () ->
                  new LongValueFacetCounts(
                      "dblm",
                      MultiDoubleValuesSource.fromDoubleField("dblm").toSortableMultiLongValuesSource(),
                      fc));
          longValues(
              facets, index, query, "lv:longm@int",
              () -> new LongValueFacetCounts("longm", MultiLongValuesSource.fromIntField("longm"), fc));

          // StringValueFacetCounts.
          for (String field : new String[] {"str", "strm", "strhi", "strhim", "nofield"}) {
            StringDocValuesReaderState state = new StringDocValuesReaderState(reader, field);
            StringValueFacetCounts f = new StringValueFacetCounts(state, fc);
            stringValues(facets, index, query, "sv:" + field, f, field);
          }

          // LongRangeFacetCounts.
          for (String set : new String[] {"excl", "gaps", "overl", "empty"}) {
            LongRange[] lr = longRanges(set);
            String p = "lr:" + set + ":";
            ranges(facets, index, query, p + "long1@field", new LongRangeFacetCounts("long1", fc, lr), "long1");
            ranges(facets, index, query, p + "longm@field", new LongRangeFacetCounts("longm", fc, lr), "longm");
            ranges(
                facets, index, query, p + "long1@src",
                new LongRangeFacetCounts("long1", LongValuesSource.fromLongField("long1"), fc, lr),
                "long1");
            ranges(
                facets, index, query, p + "longm@multi",
                new LongRangeFacetCounts("longm", MultiLongValuesSource.fromLongField("longm"), fc, lr),
                "longm");
            ranges(
                facets, index, query, p + "long1@single",
                new LongRangeFacetCounts(
                    "long1",
                    MultiLongValuesSource.fromSingleValued(LongValuesSource.fromLongField("long1")),
                    fc,
                    lr),
                "long1");
            ranges(
                facets, index, query, p + "nofield@field",
                new LongRangeFacetCounts("nofield", fc, lr), "nofield");
            if (query.equals("all")) {
              ranges(
                  facets, index, "fast", p + "long1@field",
                  new LongRangeFacetCounts("long1", (LongValuesSource) null, fcAll, FAST, lr),
                  "long1");
              ranges(
                  facets, index, "fast", p + "longm@field",
                  new LongRangeFacetCounts("longm", (MultiLongValuesSource) null, fcAll, FAST, lr),
                  "longm");
              ranges(
                  facets, index, "fast", p + "long1@src",
                  new LongRangeFacetCounts("long1", LongValuesSource.fromLongField("long1"), fcAll, FAST, lr),
                  "long1");
              ranges(
                  facets, index, "fast", p + "longm@multi",
                  new LongRangeFacetCounts(
                      "longm", MultiLongValuesSource.fromLongField("longm"), fcAll, FAST, lr),
                  "longm");
              ranges(
                  facets, index, "nomatch", p + "long1@src",
                  new LongRangeFacetCounts(
                      "long1",
                      LongValuesSource.fromLongField("long1"),
                      fcAll,
                      new TermQuery(new Term("q", "zzz")),
                      lr),
                  "long1");
            }
          }

          // DoubleRangeFacetCounts.
          for (String set : new String[] {"excl", "overl"}) {
            DoubleRange[] dr = doubleRanges(set);
            String p = "dr:" + set + ":";
            ranges(facets, index, query, p + "dbl@field", new DoubleRangeFacetCounts("dbl", fc, dr), "dbl");
            ranges(facets, index, query, p + "dblm@field", new DoubleRangeFacetCounts("dblm", fc, dr), "dblm");
            ranges(
                facets, index, query, p + "dbl@src",
                new DoubleRangeFacetCounts("dbl", DoubleValuesSource.fromDoubleField("dbl"), fc, dr),
                "dbl");
            ranges(
                facets, index, query, p + "flt@src",
                new DoubleRangeFacetCounts("flt", DoubleValuesSource.fromFloatField("flt"), fc, dr),
                "flt");
            ranges(
                facets, index, query, p + "dblm@multi",
                new DoubleRangeFacetCounts("dblm", MultiDoubleValuesSource.fromDoubleField("dblm"), fc, dr),
                "dblm");
            ranges(
                facets, index, query, p + "flt@multi",
                new DoubleRangeFacetCounts("flt", MultiDoubleValuesSource.fromFloatField("flt"), fc, dr),
                "flt");
            ranges(
                facets, index, query, p + "longm@todouble",
                new DoubleRangeFacetCounts(
                    "longm", MultiLongValuesSource.fromLongField("longm").toMultiDoubleValuesSource(), fc, dr),
                "longm");
            ranges(
                facets, index, query, p + "longm@dmulti",
                new DoubleRangeFacetCounts("longm", MultiDoubleValuesSource.fromLongField("longm"), fc, dr),
                "longm");
            ranges(
                facets, index, query, p + "dbl@single",
                new DoubleRangeFacetCounts(
                    "dbl",
                    MultiDoubleValuesSource.fromSingleValued(DoubleValuesSource.fromDoubleField("dbl")),
                    fc,
                    dr),
                "dbl");
            if (query.equals("all")) {
              ranges(
                  facets, index, "fast", p + "dbl@field",
                  new DoubleRangeFacetCounts("dbl", (DoubleValuesSource) null, fcAll, FAST, dr),
                  "dbl");
              ranges(
                  facets, index, "fast", p + "dblm@field",
                  new DoubleRangeFacetCounts("dblm", (MultiDoubleValuesSource) null, fcAll, FAST, dr),
                  "dblm");
              ranges(
                  facets, index, "fast", p + "dbl@src",
                  new DoubleRangeFacetCounts("dbl", DoubleValuesSource.fromDoubleField("dbl"), fcAll, FAST, dr),
                  "dbl");
              ranges(
                  facets, index, "fast", p + "dblm@multi",
                  new DoubleRangeFacetCounts(
                      "dblm", MultiDoubleValuesSource.fromDoubleField("dblm"), fcAll, FAST, dr),
                  "dblm");
            }
          }

          // Range-on-range.
          for (RangeFieldQuery.QueryType type : RangeFieldQuery.QueryType.values()) {
            String p = "ror:" + type + ":";
            org.apache.lucene.facet.rangeonrange.LongRange[] l1 = {
              new org.apache.lucene.facet.rangeonrange.LongRange("a", 0, true, 5, true),
              new org.apache.lucene.facet.rangeonrange.LongRange("b", -3, false, 2, true),
              new org.apache.lucene.facet.rangeonrange.LongRange("c", 10, true, 100, false),
              new org.apache.lucene.facet.rangeonrange.LongRange(
                  "every", Long.MIN_VALUE, true, Long.MAX_VALUE, true),
              new org.apache.lucene.facet.rangeonrange.LongRange("pt", 1, true, 1, true),
            };
            org.apache.lucene.facet.rangeonrange.LongRange[] l2 = {
              new org.apache.lucene.facet.rangeonrange.LongRange(
                  "q", new long[] {0, 0}, new long[] {5, 5}),
              new org.apache.lucene.facet.rangeonrange.LongRange(
                  "r", new long[] {3, 1}, new long[] {4, 2}),
              new org.apache.lucene.facet.rangeonrange.LongRange(
                  "s", new long[] {-10, -10}, new long[] {20, 20}),
            };
            org.apache.lucene.facet.rangeonrange.DoubleRange[] d1 = {
              new org.apache.lucene.facet.rangeonrange.DoubleRange("a", -1.0, true, 1.0, true),
              new org.apache.lucene.facet.rangeonrange.DoubleRange("b", 0.0, false, 2.5, false),
              new org.apache.lucene.facet.rangeonrange.DoubleRange(
                  "c", Double.NEGATIVE_INFINITY, true, Double.POSITIVE_INFINITY, true),
              new org.apache.lucene.facet.rangeonrange.DoubleRange("d", 3.0, true, 3.0, true),
            };
            ranges2(facets, index, query, p + "rorlong", new LongRangeOnRangeFacetCounts("rorlong", fc, type, l1), "rorlong");
            ranges2(facets, index, query, p + "rorlong2", new LongRangeOnRangeFacetCounts("rorlong2", fc, type, l2), "rorlong2");
            ranges2(facets, index, query, p + "rordbl", new DoubleRangeOnRangeFacetCounts("rordbl", fc, type, d1), "rordbl");
            ranges2(facets, index, query, p + "nofield", new LongRangeOnRangeFacetCounts("nofield", fc, type, l1), "nofield");
            if (query.equals("all")) {
              ranges2(
                  facets, index, "fast", p + "rorlong",
                  new LongRangeOnRangeFacetCounts("rorlong", fcAll, type, FAST, l1), "rorlong");
              ranges2(
                  facets, index, "fast", p + "rordbl",
                  new DoubleRangeOnRangeFacetCounts("rordbl", fcAll, type, FAST, d1), "rordbl");
            }
          }

          // Facet sets.
          for (String field : new String[] {"fs_long", "fs_int", "fs_dbl", "fs_flt"}) {
            MatchingFacetSetsCounts f =
                new MatchingFacetSetsCounts(field, fc, decoder(field), matchers(field));
            facetSets(facets, index, query, "fs:" + field, f, field);
            if (query.equals("all")) {
              MatchingFacetSetsCounts fast =
                  new MatchingFacetSetsCounts(field, fcAll, decoder(field), FAST, matchers(field));
              facetSets(facets, index, "fast", "fs:" + field, fast, field);
            }
          }

          // DynamicRangeUtil.
          for (int topN : new int[] {0, 1, 3, 5, 100}) {
            String r =
                attemptDynamic(
                    () ->
                        DynamicRangeUtil.computeDynamicRanges(
                            "w",
                            LongValuesSource.fromLongField("w"),
                            LongValuesSource.fromLongField("long1"),
                            fc,
                            topN,
                            exec));
            dynamic.append(index).append('\t').append(query).append("\tw:long1:").append(topN);
            dynamic.append('\t').append(r).append('\n');
          }
          for (String wf : new String[] {"wx", "wy", "nofield"}) {
            String r =
                attemptDynamic(
                    () ->
                        DynamicRangeUtil.computeDynamicRanges(
                            wf,
                            LongValuesSource.fromLongField(wf),
                            LongValuesSource.fromLongField(wf.equals("wy") ? "long1" : "w"),
                            fc,
                            4,
                            exec));
            dynamic.append(index).append('\t').append(query).append('\t').append(wf).append(":4");
            dynamic.append('\t').append(r).append('\n');
          }
        }

        // Range.getQuery.
        LongRange lq = new LongRange("lq", 0, true, 10, true);
        DoubleRange dq = new DoubleRange("dq", -1.0, false, 2.0, true);
        Map<String, Query> qs = new LinkedHashMap<>();
        qs.put("long:src", lq.getQuery(null, LongValuesSource.fromLongField("long1")));
        qs.put("long:fast", lq.getQuery(FAST, LongValuesSource.fromLongField("long1")));
        qs.put("long:nomatch", lq.getQuery(new TermQuery(new Term("q", "zzz")), LongValuesSource.fromLongField("long1")));
        qs.put("long:multi", lq.getQuery(null, MultiLongValuesSource.fromLongField("longm")));
        qs.put("long:multifast", lq.getQuery(FAST, MultiLongValuesSource.fromLongField("longm")));
        qs.put(
            "long:single",
            lq.getQuery(null, MultiLongValuesSource.fromSingleValued(LongValuesSource.fromLongField("long1"))));
        qs.put("double:src", dq.getQuery(null, DoubleValuesSource.fromDoubleField("dbl")));
        qs.put("double:fast", dq.getQuery(FAST, DoubleValuesSource.fromDoubleField("dbl")));
        qs.put("double:multi", dq.getQuery(null, MultiDoubleValuesSource.fromDoubleField("dblm")));
        qs.put("double:multifast", dq.getQuery(FAST, MultiDoubleValuesSource.fromDoubleField("dblm")));
        qs.put(
            "double:single",
            dq.getQuery(null, MultiDoubleValuesSource.fromSingleValued(DoubleValuesSource.fromFloatField("flt"))));
        for (Map.Entry<String, Query> e : qs.entrySet()) {
          FacetsCollector fc = searcher.search(e.getValue(), new FacetsCollectorManager());
          queries.append(index).append('\t').append(e.getKey()).append('\t');
          queries.append(esc(e.getValue().toString())).append('\t').append(digest(fc)).append('\n');
        }
      }

      // Hand-built DynamicRangeUtil inputs.
      Map<String, long[][]> arrays = new LinkedHashMap<>();
      arrays.put("ties", new long[][] {{5, 3, 5, 1, 3, 5, 9}, {2, 1, 1, 4, 1, 3, 2}});
      arrays.put("neg", new long[][] {{-4, 8, 0, 2}, {3, -1, 5, 2}});
      arrays.put("wrap", new long[][] {{Long.MAX_VALUE, Long.MAX_VALUE, 1}, {1, 1, 1}});
      arrays.put("one", new long[][] {{42}, {7}});
      for (Map.Entry<String, long[][]> e : arrays.entrySet()) {
        long[] values = e.getValue()[0];
        long[] weights = e.getValue()[1];
        long total = 0;
        for (long w : weights) total += w;
        for (int len : new int[] {0, values.length - 1, values.length}) {
          if (len < 0) continue;
          for (int topN : new int[] {0, 1, 2, 3, 10}) {
            long[] v = values.clone();
            long[] w = weights.clone();
            long t = 0;
            for (int k = 0; k < len; k++) t += w[k];
            String r = ranges(DynamicRangeUtil.computeDynamicNumericRanges(v, w, len, t, topN));
            dynamic.append("arrays\t").append(e.getKey()).append('\t').append(len).append(':').append(topN);
            dynamic.append('\t').append(r).append('\n');
          }
        }
      }

      // Refusals.
      error(errors, "longrange:minexcl", () -> new LongRange("x", Long.MAX_VALUE, false, 5, true));
      error(errors, "longrange:maxexcl", () -> new LongRange("x", 0, true, Long.MIN_VALUE, false));
      error(errors, "longrange:empty", () -> new LongRange("x", 5, true, 5, false));
      error(errors, "longrange:ok", () -> new LongRange("x", 5, false, 7, false));
      error(errors, "doublerange:minnan", () -> new DoubleRange("x", Double.NaN, true, 5, true));
      error(errors, "doublerange:maxnan", () -> new DoubleRange("x", 0, true, Double.NaN, true));
      error(errors, "doublerange:empty", () -> new DoubleRange("x", 1.0, false, 1.0, true));
      error(errors, "ror:long:minexcl", () -> new org.apache.lucene.facet.rangeonrange.LongRange("x", Long.MAX_VALUE, false, 5, true));
      error(errors, "ror:long:maxexcl", () -> new org.apache.lucene.facet.rangeonrange.LongRange("x", 0, true, Long.MIN_VALUE, false));
      error(errors, "ror:long:inverted", () -> new org.apache.lucene.facet.rangeonrange.LongRange("x", 5, true, 4, true));
      error(errors, "ror:long:emptydims", () -> new org.apache.lucene.facet.rangeonrange.LongRange("x", new long[0], new long[0]));
      error(errors, "ror:long:mismatch", () -> new org.apache.lucene.facet.rangeonrange.LongRange("x", new long[] {1}, new long[] {2, 3}));
      error(errors, "ror:long:invertdim", () -> new org.apache.lucene.facet.rangeonrange.LongRange("x", new long[] {1, 5}, new long[] {2, 3}));
      error(errors, "ror:double:minnan", () -> new org.apache.lucene.facet.rangeonrange.DoubleRange("x", Double.NaN, true, 1, true));
      error(errors, "ror:double:maxnan", () -> new org.apache.lucene.facet.rangeonrange.DoubleRange("x", 0, true, Double.NaN, true));
      error(errors, "ror:double:inverted", () -> new org.apache.lucene.facet.rangeonrange.DoubleRange("x", 1, false, 1, true));
      error(errors, "ror:double:emptydims", () -> new org.apache.lucene.facet.rangeonrange.DoubleRange("x", new double[0], new double[0]));
      error(errors, "ror:double:mismatch", () -> new org.apache.lucene.facet.rangeonrange.DoubleRange("x", new double[] {1}, new double[0]));
      error(errors, "ror:double:invertdim", () -> new org.apache.lucene.facet.rangeonrange.DoubleRange("x", new double[] {3}, new double[] {2}));
      IndexSearcher ms = new IndexSearcher(multi);
      FacetsCollector all = ms.search(new MatchAllDocsQuery(), new FacetsCollectorManager());
      error(
          errors,
          "ror:double:nanarray",
          () ->
              new DoubleRangeOnRangeFacetCounts(
                  "rordbl",
                  all,
                  RangeFieldQuery.QueryType.INTERSECTS,
                  new org.apache.lucene.facet.rangeonrange.DoubleRange(
                      "x", new double[] {Double.NaN}, new double[] {1})));
      error(
          errors,
          "ror:double:nanmax",
          () ->
              new DoubleRangeOnRangeFacetCounts(
                  "rordbl",
                  all,
                  RangeFieldQuery.QueryType.INTERSECTS,
                  new org.apache.lucene.facet.rangeonrange.DoubleRange(
                      "x", new double[] {1}, new double[] {Double.NaN})));
      error(errors, "fs:long:empty", () -> new LongFacetSet());
      error(errors, "fs:int:empty", () -> new IntFacetSet());
      error(errors, "fs:float:empty", () -> new FloatFacetSet());
      error(errors, "fs:double:empty", () -> new DoubleFacetSet());
      error(errors, "fs:field:empty", () -> FacetSetsField.create("f"));
      error(
          errors,
          "fs:field:dims",
          () -> FacetSetsField.create("f", new LongFacetSet(1, 2), new LongFacetSet(1)));
      error(
          errors,
          "fs:field:type",
          () -> FacetSetsField.create("f", new LongFacetSet(1), new DoubleFacetSet(1)));
      error(errors, "fs:dim:minexcl", () -> DimRange.fromLongs(Long.MAX_VALUE, false, 1, true));
      error(errors, "fs:dim:maxexcl", () -> DimRange.fromLongs(1, true, Long.MIN_VALUE, false));
      error(errors, "fs:dim:inverted", () -> DimRange.fromLongs(5, true, 5, false));
      error(errors, "fs:dim:dminnan", () -> DimRange.fromDoubles(Double.NaN, true, 1, true));
      error(errors, "fs:dim:dmaxnan", () -> DimRange.fromDoubles(1, true, Double.NaN, true));
      error(errors, "fs:dim:dinverted", () -> DimRange.fromDoubles(1, false, 1, true));
      error(errors, "fs:dim:fminnan", () -> DimRange.fromFloats(Float.NaN, true, 1, true));
      error(errors, "fs:dim:fmaxnan", () -> DimRange.fromFloats(1, true, Float.NaN, true));
      error(errors, "fs:dim:finverted", () -> DimRange.fromFloats(1, true, 1, false));
      error(errors, "fs:dim:dvalues", () -> DimRange.fromDoubles(-0.0, true, 0.0, false));
      error(errors, "fs:dim:fvalues", () -> DimRange.fromFloats(-1.5f, false, 2.5f, true));
      error(errors, "fs:matcher:empty", () -> new RangeFacetSetMatcher("x"));
      error(
          errors,
          "fs:counts:nomatchers",
          () -> new MatchingFacetSetsCounts("fs_long", all, FacetSetDecoder::decodeLongs));
      error(
          errors,
          "fs:counts:dims",
          () ->
              new MatchingFacetSetsCounts(
                  "fs_long",
                  all,
                  FacetSetDecoder::decodeLongs,
                  new ExactFacetSetMatcher("a", new LongFacetSet(1, 2)),
                  new ExactFacetSetMatcher("b", new LongFacetSet(1))));
      error(
          errors,
          "sv:state",
          () -> {
            StringDocValuesReaderState state = new StringDocValuesReaderState(single, "str");
            return new StringValueFacetCounts(state, all);
          });
      error(
          errors,
          "sv:state:multi",
          () -> {
            DirectoryReader other = DirectoryReader.open(md);
            try {
              StringDocValuesReaderState state = new StringDocValuesReaderState(other, "str");
              return new StringValueFacetCounts(state, all);
            } finally {
              other.close();
            }
          });
      error(
          errors,
          "lv:wrongtype",
          () -> new LongValueFacetCounts("str", all));
      error(
          errors,
          "sv:wrongtype",
          () -> new StringValueFacetCounts(new StringDocValuesReaderState(multi, "long1"), all));
    } finally {
      exec.shutdown();
    }
    Files.writeString(root.resolve("facets.tsv"), facets.toString(), StandardCharsets.UTF_8);
    Files.writeString(root.resolve("queries.tsv"), queries.toString(), StandardCharsets.UTF_8);
    Files.writeString(root.resolve("dynamic.tsv"), dynamic.toString(), StandardCharsets.UTF_8);
    Files.writeString(root.resolve("errors.tsv"), errors.toString(), StandardCharsets.UTF_8);
  }

  static void stringValues(
      StringBuilder sb, String index, String query, String name, StringValueFacetCounts f, String dim) {
    line(sb, index, query, name + " top1", attempt(() -> f.getTopChildren(1, dim)));
    line(sb, index, query, name + " top3", attempt(() -> f.getTopChildren(3, dim)));
    line(sb, index, query, name + " top100", attempt(() -> f.getTopChildren(100, dim)));
    line(sb, index, query, name + " top5000", attempt(() -> f.getTopChildren(5000, dim)));
    line(sb, index, query, name + " dims2", attempt(() -> list(f.getAllDims(2))));
    line(sb, index, query, name + " top0", attempt(() -> f.getTopChildren(0, dim)));
    line(sb, index, query, name + " baddim", attempt(() -> f.getTopChildren(3, "nope")));
    line(sb, index, query, name + " badpath", attempt(() -> f.getTopChildren(3, dim, "x")));
    line(sb, index, query, name + " allbaddim", attempt(() -> f.getAllChildren("nope")));
    line(sb, index, query, name + " all", attempt(() -> res(f.getAllChildren(dim), true)));
    for (String label : new String[] {"s3", "s22", "zzz", "m1", "h0007", "x0001", "y3", "été"}) {
      line(sb, index, query, name + " spec:" + esc(label), attempt(() -> f.getSpecificValue(dim, label)));
    }
    line(sb, index, query, name + " specbaddim", attempt(() -> f.getSpecificValue("nope", "s1")));
    line(sb, index, query, name + " specnopath", attempt(() -> f.getSpecificValue(dim)));
  }

  static void ranges2(StringBuilder sb, String index, String query, String name, Facets f, String dim) {
    standard(sb, index, query, name, f, dim);
    line(sb, index, query, name + " all", attempt(() -> f.getAllChildren(dim)));
  }

  static void facetSets(StringBuilder sb, String index, String query, String name, Facets f, String dim) {
    line(sb, index, query, name + " top1", attempt(() -> f.getTopChildren(1, dim)));
    line(sb, index, query, name + " top2", attempt(() -> f.getTopChildren(2, dim)));
    line(sb, index, query, name + " top100", attempt(() -> f.getTopChildren(100, dim)));
    line(sb, index, query, name + " topother", attempt(() -> f.getTopChildren(2, "other")));
    line(sb, index, query, name + " dims2", attempt(() -> list(f.getAllDims(2))));
    line(sb, index, query, name + " top0", attempt(() -> f.getTopChildren(0, dim)));
    line(sb, index, query, name + " all", attempt(() -> f.getAllChildren(dim)));
    line(sb, index, query, name + " allbaddim", attempt(() -> f.getAllChildren("nope")));
    line(sb, index, query, name + " allbadpath", attempt(() -> f.getAllChildren(dim, "x")));
    line(sb, index, query, name + " spec", attempt(() -> f.getSpecificValue(dim, "x")));
  }

  static String attemptDynamic(Call c) {
    try {
      @SuppressWarnings("unchecked")
      List<DynamicRangeUtil.DynamicRangeInfo> l = (List<DynamicRangeUtil.DynamicRangeInfo>) c.run();
      return l == null ? "null" : ranges(l);
    } catch (IllegalArgumentException e) {
      return "IAE:" + esc(e.getMessage());
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  static void error(StringBuilder sb, String name, Call c) {
    String r = attempt(c);
    if (!(r.startsWith("IAE:") || r.startsWith("ISE:") || r.equals("UOE"))) {
      r = "ok:" + r;
    }
    sb.append(name).append('\t').append(r).append('\n');
  }
}
