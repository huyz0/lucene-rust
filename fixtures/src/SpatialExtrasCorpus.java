import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.IndexableFieldType;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.spatial.SpatialStrategy;
import org.apache.lucene.spatial.bbox.BBoxOverlapRatioValueSource;
import org.apache.lucene.spatial.bbox.BBoxStrategy;
import org.apache.lucene.spatial.composite.CompositeSpatialStrategy;
import org.apache.lucene.spatial.prefix.HeatmapFacetCounter;
import org.apache.lucene.spatial.prefix.NumberRangePrefixTreeStrategy;
import org.apache.lucene.spatial.prefix.PrefixTreeStrategy;
import org.apache.lucene.spatial.prefix.RecursivePrefixTreeStrategy;
import org.apache.lucene.spatial.prefix.TermQueryPrefixTreeStrategy;
import org.apache.lucene.spatial.prefix.tree.DateRangePrefixTree;
import org.apache.lucene.spatial.prefix.tree.GeohashPrefixTree;
import org.apache.lucene.spatial.prefix.tree.NumberRangePrefixTree.UnitNRShape;
import org.apache.lucene.spatial.prefix.tree.PackedQuadPrefixTree;
import org.apache.lucene.spatial.prefix.tree.QuadPrefixTree;
import org.apache.lucene.spatial.prefix.tree.S2PrefixTree;
import org.apache.lucene.spatial.query.SpatialArgs;
import org.apache.lucene.spatial.query.SpatialOperation;
import org.apache.lucene.spatial.serialized.SerializedDVStrategy;
import org.apache.lucene.spatial.util.CachingDoubleValueSource;
import org.apache.lucene.spatial.util.ShapeAreaValueSource;
import org.apache.lucene.spatial.vector.PointVectorStrategy;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.locationtech.spatial4j.context.SpatialContext;
import org.locationtech.spatial4j.context.SpatialContextFactory;
import org.locationtech.spatial4j.shape.Point;
import org.locationtech.spatial4j.shape.Rectangle;
import org.locationtech.spatial4j.shape.Shape;

/**
 * The spatial-extras strategies corpus (M9 T9.5), shared by {@code GenSpatialStrategies} (which
 * indexes it and records Lucene's answers) and {@code VerifySpatialExtras} (which replays the
 * answers over the index this port writes): the contexts, the strategies, how a document's shape
 * specs become fields, how a field is described, and how every recorded question is answered.
 *
 * <p>A shape spec is {@code <family>:<text>}: family {@code g} (geodetic, Spatial4j's {@code
 * SpatialContext.GEO}), {@code t} (Geo3D on the sphere), {@code f} (planar, +-1000) or {@code p} (a
 * geodetic point) with WKT text, or {@code d} with a {@code DateRangePrefixTree} string.
 */
public class SpatialExtrasCorpus {
  static final SpatialContext GEO = SpatialContext.GEO;
  static final SpatialContext G3 = ctx("spatialContextFactory", "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory");
  static final SpatialContext FLAT = ctx("geo", "false", "worldBounds", "ENVELOPE(-1000, 1000, 1000, -1000)");
  static final DateRangePrefixTree DATES = new DateRangePrefixTree(DateRangePrefixTree.DEFAULT_CAL);

  static SpatialContext ctx(String... kv) {
    Map<String, String> m = new LinkedHashMap<>();
    for (int i = 0; i < kv.length; i += 2) m.put(kv[i], kv[i + 1]);
    return SpatialContextFactory.makeSpatialContext(m, SpatialExtrasCorpus.class.getClassLoader());
  }

  /** Strategy name, the shape family it indexes ({@code -} for none: query only). */
  static final String[][] STRATEGIES = {
    {"rgh", "g"}, {"rq", "g"}, {"rqn", "g"}, {"rpq", "g"}, {"rs2", "t"}, {"rfl", "f"}, {"pts", "p"},
    {"tq", "g"}, {"bb", "g"}, {"bbf", "f"}, {"pv", "p"}, {"sdv", "g"}, {"sd3", "t"}, {"cmp", "g"},
    {"cmpn", "-"}, {"dr", "d"},
  };

  static final Map<String, SpatialStrategy> S = new LinkedHashMap<>();

  static {
    S.put("rgh", new RecursivePrefixTreeStrategy(new GeohashPrefixTree(GEO, 4), "rgh"));
    S.put("rq", new RecursivePrefixTreeStrategy(new QuadPrefixTree(GEO, 8), "rq"));
    RecursivePrefixTreeStrategy rqn = new RecursivePrefixTreeStrategy(new QuadPrefixTree(GEO, 8), "rqn");
    rqn.setPruneLeafyBranches(false);
    rqn.setPrefixGridScanLevel(3);
    rqn.setMultiOverlappingIndexedShapes(false);
    rqn.setDistErrPct(0.1);
    S.put("rqn", rqn);
    S.put("rpq", new RecursivePrefixTreeStrategy(new PackedQuadPrefixTree(GEO, 10), "rpq"));
    S.put("rs2", new RecursivePrefixTreeStrategy(new S2PrefixTree(G3, 5, 1), "rs2"));
    S.put("rfl", new RecursivePrefixTreeStrategy(new QuadPrefixTree(FLAT, 9), "rfl"));
    RecursivePrefixTreeStrategy pts = new RecursivePrefixTreeStrategy(new QuadPrefixTree(GEO, 10), "pts");
    pts.setPointsOnly(true);
    S.put("pts", pts);
    S.put("tq", new TermQueryPrefixTreeStrategy(new GeohashPrefixTree(GEO, 5), "tq"));
    S.put("bb", BBoxStrategy.newInstance(GEO, "bb"));
    FieldType all = new FieldType(BBoxStrategy.DEFAULT_FIELDTYPE);
    all.setStored(true);
    S.put("bbf", new BBoxStrategy(FLAT, "bbf", all));
    S.put("pv", PointVectorStrategy.newInstance(GEO, "pv"));
    S.put("sdv", new SerializedDVStrategy(GEO, "sdv"));
    S.put("sd3", new SerializedDVStrategy(G3, "sd3"));
    CompositeSpatialStrategy cmp =
        new CompositeSpatialStrategy(
            "cmp",
            new RecursivePrefixTreeStrategy(new QuadPrefixTree(GEO, 8), "cmp_rpt"),
            new SerializedDVStrategy(GEO, "cmp_sdv"));
    S.put("cmp", cmp);
    CompositeSpatialStrategy cmpn =
        new CompositeSpatialStrategy("cmpn", cmp.getIndexStrategy(), cmp.getGeometryStrategy());
    cmpn.setOptimizePredicates(false);
    S.put("cmpn", cmpn);
    S.put("dr", new NumberRangePrefixTreeStrategy(DATES, "dr"));
  }

  static SpatialContext contextOf(String family) {
    switch (family) {
      case "t":
        return G3;
      case "f":
        return FLAT;
      default:
        return GEO;
    }
  }

  /** A spec's shape. */
  static Shape shape(String spec) throws Exception {
    String family = spec.substring(0, spec.indexOf(':'));
    String text = spec.substring(spec.indexOf(':') + 1);
    if (family.equals("d")) return DATES.parseShape(text);
    return contextOf(family).getFormats().getWktReader().read(text);
  }

  static String esc(String s) {
    return s == null ? "null" : s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n");
  }

  static String h(double v) {
    return Long.toHexString(Double.doubleToRawLongBits(v));
  }

  static String hex(BytesRef b) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < b.length; i++) sb.append(String.format("%02x", b.bytes[b.offset + i]));
    return sb.toString();
  }

  static String err(Throwable e) {
    return "E\t" + e.getClass().getName() + "\t" + esc(e.getMessage());
  }

  /** How a field looks to the indexing chain: name, type, tokens, numeric/binary/string values. */
  static String describe(IndexableField f) throws IOException {
    IndexableFieldType t = f.fieldType();
    StringBuilder sb = new StringBuilder(f.name()).append('|');
    sb.append(t.stored() ? 's' : '-').append(t.tokenized() ? 't' : '-').append(t.omitNorms() ? 'n' : '-');
    sb.append('|').append(t.indexOptions().ordinal()).append('|').append(t.docValuesType().ordinal());
    sb.append('|').append(t.pointDimensionCount()).append('x').append(t.pointNumBytes());
    sb.append('|');
    if (f instanceof Field ff && ff.tokenStreamValue() != null) {
      try (TokenStream ts = ff.tokenStreamValue()) {
        TermToBytesRefAttribute term = ts.addAttribute(TermToBytesRefAttribute.class);
        ts.reset();
        List<String> tokens = new ArrayList<>();
        while (ts.incrementToken()) tokens.add(hex(term.getBytesRef()));
        ts.end();
        sb.append(tokens(tokens));
      }
    } else {
      sb.append('-');
    }
    sb.append('|');
    Number n = f.numericValue();
    if (n == null) sb.append('-');
    else if (n instanceof Double d) sb.append("D").append(h(d));
    else if (n instanceof Long l) sb.append("L").append(Long.toHexString(l));
    else sb.append(n.getClass().getSimpleName()).append(n);
    sb.append('|');
    BytesRef b = f.binaryValue();
    sb.append(b == null ? "-" : hex(b));
    sb.append('|');
    String s = f.stringValue();
    sb.append(s == null ? "-" : esc(s));
    return sb.toString();
  }

  /**
   * A token list: all of them up to {@value #TOKENS_SHOWN}; past that the count, the first three
   * and an FNV-1a hash of every token's bytes with a 0xff separator, which still pins the whole
   * sequence.
   */
  static String tokens(List<String> tokens) {
    if (tokens.size() <= TOKENS_SHOWN) return String.join(",", tokens);
    long hash = 0xcbf29ce484222325L;
    for (String t : tokens) {
      for (int i = 0; i < t.length(); i += 2) {
        hash ^= Integer.parseInt(t.substring(i, i + 2), 16);
        hash *= 0x100000001b3L;
      }
      hash ^= 0xff;
      hash *= 0x100000001b3L;
    }
    return "#" + tokens.size() + ":" + String.join(",", tokens.subList(0, 3)) + ":" + Long.toHexString(hash);
  }

  static final int TOKENS_SHOWN = 24;

  /** The fields one strategy makes of one shape spec. */
  static Field[] fields(String strategy, String spec) throws Exception {
    return S.get(strategy).createIndexableFields(shape(spec));
  }

  /** The document of {@code docs.tsv}'s line: {@code id} then {@code strategy=spec} fields. */
  static Document document(String line) throws Exception {
    String[] p = line.split("\t");
    Document doc = new Document();
    doc.add(new StringField("id", p[0], Field.Store.YES));
    for (int i = 1; i < p.length; i++) {
      int eq = p[i].indexOf('=');
      for (Field f : fields(p[i].substring(0, eq), p[i].substring(eq + 1))) doc.add(f);
    }
    return doc;
  }

  static String hexBits(int maxDoc, ScoreDoc[] docs) {
    byte[] b = new byte[(maxDoc + 7) / 8];
    for (ScoreDoc sd : docs) b[sd.doc >> 3] |= (byte) (1 << (sd.doc & 7));
    int end = b.length;
    while (end > 0 && b[end - 1] == 0) end--;
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < end; i++) sb.append(String.format("%02x", b[i] & 0xff));
    return sb.toString();
  }

  /** Global docs whose id is not a multiple of three: the heatmaps' {@code topAcceptDocs}. */
  static Bits mask(IndexReader reader) throws IOException {
    int maxDoc = reader.maxDoc();
    boolean[] keep = new boolean[maxDoc];
    for (int d = 0; d < maxDoc; d++) {
      keep[d] = Integer.parseInt(reader.storedFields().document(d).get("id")) % 3 != 0;
    }
    return new Bits() {
      @Override
      public boolean get(int index) {
        return keep[index];
      }

      @Override
      public int length() {
        return maxDoc;
      }
    };
  }

  /** The answer to one recorded question (a {@code queries.tsv} line's inputs). */
  static String answer(IndexSearcher s, String[] a) {
    try {
      return answerOrThrow(s, a);
    } catch (Exception | AssertionError e) {
      return err(e);
    }
  }

  static String answerOrThrow(IndexSearcher s, String[] a) throws Exception {
    IndexReader reader = s.getIndexReader();
    SpatialStrategy st = S.get(a[1]);
    switch (a[0]) {
      case "s":
        return esc(st.toString().replaceAll("@[0-9a-f]+", ""));
      case "q":
        {
          SpatialArgs args = new SpatialArgs(SpatialOperation.get(a[2]), shape(a[3]));
          if (!a[4].equals("-")) args.setDistErrPct(Double.parseDouble(a[4]));
          Query q = st.makeQuery(args);
          TopDocs td = s.search(q, Math.max(1, reader.maxDoc()));
          boolean constant = true;
          for (ScoreDoc sd : td.scoreDocs) constant &= sd.score == 1f;
          if (constant) {
            return "C\t" + td.scoreDocs.length + "\t" + hexBits(reader.maxDoc(), td.scoreDocs);
          }
          // RPT's points-only point query is a scored TermQuery
          StringBuilder sb = new StringBuilder("S\t").append(td.scoreDocs.length).append('\t');
          ScoreDoc[] sorted = td.scoreDocs.clone();
          java.util.Arrays.sort(sorted, (x, y) -> Integer.compare(x.doc, y.doc));
          for (int i = 0; i < sorted.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(sorted[i].doc).append(':').append(Integer.toHexString(Float.floatToRawIntBits(sorted[i].score)));
          }
          return sb.toString();
        }
      case "v":
        return values(reader, valueSource(st, a));
      case "h":
        {
          Shape shape = a[2].equals("-") ? null : shape(a[2]);
          Bits accept = a[5].equals("mask") ? mask(reader) : null;
          HeatmapFacetCounter.Heatmap hm =
              HeatmapFacetCounter.calcFacets(
                  (PrefixTreeStrategy) st,
                  reader.getContext(),
                  accept,
                  shape,
                  Integer.parseInt(a[3]),
                  Integer.parseInt(a[4]));
          StringBuilder sb =
              new StringBuilder("H\t").append(hm.columns).append('\t').append(hm.rows).append('\t');
          sb.append(esc(hm.region.toString())).append('\t');
          for (int i = 0; i < hm.counts.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(hm.counts[i]);
          }
          return sb.toString();
        }
      case "f":
        {
          NumberRangePrefixTreeStrategy nr = (NumberRangePrefixTreeStrategy) st;
          Bits accept = a[4].equals("mask") ? mask(reader) : null;
          return "F\t"
              + esc(
                  nr.calcFacets(
                          reader.getContext(),
                          accept,
                          (UnitNRShape) shape(a[2]),
                          (UnitNRShape) shape(a[3]))
                      .toString());
        }
      case "fr":
        {
          NumberRangePrefixTreeStrategy nr = (NumberRangePrefixTreeStrategy) st;
          return "F\t"
              + esc(
                  nr.calcFacets(reader.getContext(), null, shape(a[2]), Integer.parseInt(a[3]))
                      .toString());
        }
      default:
        throw new IllegalArgumentException("unknown question " + a[0]);
    }
  }

  /** {@code v strategy kind args...}: the value source a question names. */
  static DoubleValuesSource valueSource(SpatialStrategy st, String[] a) throws Exception {
    switch (a[2]) {
      case "dist":
        {
          Point p = (Point) shape(a[3]);
          return st.makeDistanceValueSource(p, Double.parseDouble(a[4]));
        }
      case "recip":
        return st.makeRecipDistanceValueSource(shape(a[3]));
      case "overlap":
        {
          BBoxStrategy bb = (BBoxStrategy) st;
          return new BBoxOverlapRatioValueSource(
              bb.makeShapeValueSource(),
              st.getSpatialContext().isGeo(),
              (Rectangle) shape(a[3]),
              Double.parseDouble(a[4]),
              Double.parseDouble(a[5]));
        }
      case "overlap2":
        return ((BBoxStrategy) st).makeOverlapRatioValueSource((Rectangle) shape(a[3]), Double.parseDouble(a[4]));
      case "area":
        {
          org.apache.lucene.spatial.ShapeValuesSource src =
              st instanceof BBoxStrategy bb
                  ? bb.makeShapeValueSource()
                  : ((SerializedDVStrategy) st).makeShapeValueSource();
          return new ShapeAreaValueSource(
              src, st.getSpatialContext(), Boolean.parseBoolean(a[3]), Double.parseDouble(a[4]));
        }
      case "cached":
        return new CachingDoubleValueSource(
            st.makeDistanceValueSource((Point) shape(a[3]), Double.parseDouble(a[4])));
      default:
        throw new IllegalArgumentException("unknown source " + a[2]);
    }
  }

  /** Every document's value (deleted ones too), in global doc order: {@code -} for none. */
  static String values(IndexReader reader, DoubleValuesSource src) throws IOException {
    StringBuilder sb = new StringBuilder("V\t");
    boolean first = true;
    for (LeafReaderContext leaf : reader.leaves()) {
      DoubleValues v = src.getValues(leaf, null);
      for (int d = 0; d < leaf.reader().maxDoc(); d++) {
        if (!first) sb.append(',');
        first = false;
        sb.append(v.advanceExact(d) ? h(v.doubleValue()) : "-");
      }
    }
    return sb.toString();
  }

  /** The {@code queries.tsv} questions, answered over one index. */
  static List<String> answerAll(IndexSearcher s, List<String[]> questions) {
    List<String> out = new ArrayList<>();
    for (String[] q : questions) out.add(answer(s, q));
    return out;
  }
}
