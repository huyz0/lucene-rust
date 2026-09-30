import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.TermFrequencyAttribute;
import org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.BinaryPoint;
import org.apache.lucene.document.DateTools;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleField;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.DoubleRange;
import org.apache.lucene.document.DoubleRangeDocValuesField;
import org.apache.lucene.document.FeatureField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.FloatField;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.document.FloatRange;
import org.apache.lucene.document.FloatRangeDocValuesField;
import org.apache.lucene.document.InetAddressPoint;
import org.apache.lucene.document.InetAddressRange;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.IntRange;
import org.apache.lucene.document.IntRangeDocValuesField;
import org.apache.lucene.document.KeywordField;
import org.apache.lucene.document.LateInteractionField;
import org.apache.lucene.document.LongField;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.LongRange;
import org.apache.lucene.document.LongRangeDocValuesField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.document.StoredValue;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * The document package: every field type's indexed bytes, and every field-level query's hits and
 * scores, recorded from Lucene for crates/lucene-search/tests/document_fields_fixtures.rs.
 *
 * <p>Documents are generated as <em>field specs</em> -- one line per field, a class name and its
 * constructor arguments -- and built from those specs, so the Rust test can build the very same
 * documents from the same lines. Written to {@code document_fields/}:
 *
 * <pre>
 *   index/        three segments of 300 documents, every 17th id deleted in segments 0 and 2
 *   docs.tsv      doc id, then one field spec per column
 *   facets.tsv    per field of every 5th document: what IndexingChain reads from it (field type,
 *                 binaryValue, numericValue, storedValue, invertableType, the token stream)
 *   queries.tsv   query spec, then its hits (see hits())
 *   sorts.tsv     sort spec, then the top 40 of a match-all query as doc:value
 *   values.tsv    FeatureField.newDoubleValues per document: doc, feature, value bits or -
 *   dates.tsv     DateTools.timeToString / round / stringToTime
 * </pre>
 *
 * <p>Two index-sorted twins exercise {@code SortedSkipperScorerSupplier}: {@code
 * document_fields_sorted_num/} (sorted by a skip-indexed NUMERIC field, descending) and {@code
 * document_fields_sorted_kw/} (by a skip-indexed SORTED field, ascending), each with its own
 * queries.tsv.
 */
public class GenDocumentFields {
  static final int SEGMENTS = 3;
  static final int DOCS_PER_SEGMENT = 300;
  static final String[] WORDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"
  };
  static final HexFormat HEX = HexFormat.of();
  static final Analyzer ANALYZER = new StandardAnalyzer();

  static String hex(BytesRef b) {
    return HEX.formatHex(b.bytes, b.offset, b.offset + b.length);
  }

  static byte[] unhex(String s) {
    return HEX.parseHex(s);
  }

  static int[] ints(String s) {
    return Arrays.stream(s.split(",")).mapToInt(Integer::parseInt).toArray();
  }

  static long[] longs(String s) {
    return Arrays.stream(s.split(",")).mapToLong(Long::parseLong).toArray();
  }

  static float[] floats(String s) {
    String[] p = s.split(",");
    float[] f = new float[p.length];
    for (int i = 0; i < p.length; i++) f[i] = Float.parseFloat(p[i]);
    return f;
  }

  static double[] doubles(String s) {
    return Arrays.stream(s.split(",")).mapToDouble(Double::parseDouble).toArray();
  }

  static byte[][] byteDims(String s) {
    String[] p = s.split(",");
    byte[][] b = new byte[p.length][];
    for (int i = 0; i < p.length; i++) b[i] = unhex(p[i]);
    return b;
  }

  static InetAddress ip(String s) {
    try {
      return InetAddress.getByName(s);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  static Field.Store store(String s) {
    return s.equals("Y") ? Field.Store.YES : Field.Store.NO;
  }

  static List<BytesRef> byteRefs(String s) {
    List<BytesRef> out = new ArrayList<>();
    for (String v : s.split(",")) out.add(new BytesRef(v));
    return out;
  }

  /** A field type with offsets, for the one {@code custom} field. */
  static final FieldType CUSTOM = new FieldType();

  static {
    CUSTOM.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
    CUSTOM.setTokenized(true);
    CUSTOM.freeze();
  }

  /** Builds one field from its spec. */
  static IndexableField field(String spec) {
    String[] a = spec.split(" ");
    switch (a[0]) {
      case "id":
        return new StringField("id", a[1], Field.Store.YES);
      case "text":
        return new TextField(a[1], a[3].replace('_', ' '), store(a[2]));
      case "custom":
        return new Field(a[1], a[2].replace('_', ' '), CUSTOM);
      case "stored":
        switch (a[2]) {
          case "int":
            return new StoredField(a[1], Integer.parseInt(a[3]));
          case "long":
            return new StoredField(a[1], Long.parseLong(a[3]));
          case "float":
            return new StoredField(a[1], Float.parseFloat(a[3]));
          case "double":
            return new StoredField(a[1], Double.parseDouble(a[3]));
          case "string":
            return new StoredField(a[1], a[3]);
          case "bytes":
            return new StoredField(a[1], unhex(a[3]));
          default:
            throw new IllegalArgumentException(spec);
        }
      case "IntField":
        return new IntField(a[1], Integer.parseInt(a[2]), store(a[3]));
      case "LongField":
        return new LongField(a[1], Long.parseLong(a[2]), store(a[3]));
      case "FloatField":
        return new FloatField(a[1], Float.parseFloat(a[2]), store(a[3]));
      case "DoubleField":
        return new DoubleField(a[1], Double.parseDouble(a[2]), store(a[3]));
      case "IntPoint":
        return new IntPoint(a[1], ints(a[2]));
      case "LongPoint":
        return new LongPoint(a[1], longs(a[2]));
      case "FloatPoint":
        return new FloatPoint(a[1], floats(a[2]));
      case "DoublePoint":
        return new DoublePoint(a[1], doubles(a[2]));
      case "BinaryPoint":
        return new BinaryPoint(a[1], byteDims(a[2]));
      case "InetAddressPoint":
        return new InetAddressPoint(a[1], ip(a[2]));
      case "IntRange":
        return new IntRange(a[1], ints(a[2]), ints(a[3]));
      case "LongRange":
        return new LongRange(a[1], longs(a[2]), longs(a[3]));
      case "FloatRange":
        return new FloatRange(a[1], floats(a[2]), floats(a[3]));
      case "DoubleRange":
        return new DoubleRange(a[1], doubles(a[2]), doubles(a[3]));
      case "InetAddressRange":
        return new InetAddressRange(a[1], ip(a[2]), ip(a[3]));
      case "IntRangeDocValuesField":
        return new IntRangeDocValuesField(a[1], ints(a[2]), ints(a[3]));
      case "LongRangeDocValuesField":
        return new LongRangeDocValuesField(a[1], longs(a[2]), longs(a[3]));
      case "FloatRangeDocValuesField":
        return new FloatRangeDocValuesField(a[1], floats(a[2]), floats(a[3]));
      case "DoubleRangeDocValuesField":
        return new DoubleRangeDocValuesField(a[1], doubles(a[2]), doubles(a[3]));
      case "KeywordField":
        return new KeywordField(a[1], a[2], store(a[3]));
      case "NumericDocValuesField":
        return a[3].equals("1")
            ? NumericDocValuesField.indexedField(a[1], Long.parseLong(a[2]))
            : new NumericDocValuesField(a[1], Long.parseLong(a[2]));
      case "SortedNumericDocValuesField":
        return a[3].equals("1")
            ? SortedNumericDocValuesField.indexedField(a[1], Long.parseLong(a[2]))
            : new SortedNumericDocValuesField(a[1], Long.parseLong(a[2]));
      case "SortedDocValuesField":
        return a[3].equals("1")
            ? SortedDocValuesField.indexedField(a[1], new BytesRef(a[2]))
            : new SortedDocValuesField(a[1], new BytesRef(a[2]));
      case "SortedSetDocValuesField":
        return a[3].equals("1")
            ? SortedSetDocValuesField.indexedField(a[1], new BytesRef(a[2]))
            : new SortedSetDocValuesField(a[1], new BytesRef(a[2]));
      case "BinaryDocValuesField":
        return new BinaryDocValuesField(a[1], new BytesRef(unhex(a[2])));
      case "FeatureField":
        return new FeatureField(a[1], a[2], Float.parseFloat(a[3]));
      case "LateInteractionField":
        {
          String[] vs = a[2].split(";");
          float[][] v = new float[vs.length][];
          for (int i = 0; i < vs.length; i++) v[i] = floats(vs[i]);
          return new LateInteractionField(a[1], v);
        }
      default:
        throw new IllegalArgumentException(spec);
    }
  }

  static String words(Random r, int min, int max) {
    int n = min + r.nextInt(max - min + 1);
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < n; i++) {
      if (i > 0) sb.append('_');
      String w = WORDS[r.nextInt(WORDS.length)];
      sb.append(r.nextInt(5) == 0 ? w.toUpperCase(java.util.Locale.ROOT) : w);
    }
    return sb.toString();
  }

  static String yn(Random r) {
    return r.nextBoolean() ? "Y" : "N";
  }

  static String randomHex(Random r, int n) {
    byte[] b = new byte[n];
    r.nextBytes(b);
    return HEX.formatHex(b);
  }

  static String addr(Random r) {
    return r.nextInt(10) < 7
        ? "10.0." + r.nextInt(4) + "." + r.nextInt(256)
        : "2001:db8::" + Integer.toHexString(r.nextInt(3)) + ":" + Integer.toHexString(r.nextInt(65536));
  }

  /** One document's field specs. */
  static List<String> docSpecs(Random r, int id) {
    List<String> f = new ArrayList<>();
    f.add("id " + id);
    if (r.nextInt(10) < 7) f.add("text body " + yn(r) + " " + words(r, 1, 8));
    if (r.nextInt(10) < 2) f.add("custom cust " + words(r, 2, 5));
    if (r.nextInt(10) < 3) {
      switch (r.nextInt(6)) {
        case 0 -> f.add("stored st int " + (r.nextInt(2001) - 1000));
        case 1 -> f.add("stored st long " + r.nextLong());
        case 2 -> f.add("stored st float " + (r.nextInt(401) - 200) / 8f);
        case 3 -> f.add("stored st double " + (r.nextInt(401) - 200) / 16.0);
        case 4 -> f.add("stored st string " + WORDS[r.nextInt(WORDS.length)]);
        default -> f.add("stored st bytes " + randomHex(r, 1 + r.nextInt(6)));
      }
    }
    if (r.nextInt(10) < 9) {
      f.add("IntField i " + (r.nextInt(101) - 50) + " " + yn(r));
      if (r.nextInt(5) == 0) f.add("IntField i " + (r.nextInt(101) - 50) + " N");
    }
    if (r.nextInt(10) < 9) {
      long v =
          r.nextInt(50) == 0
              ? (r.nextBoolean() ? Long.MIN_VALUE : Long.MAX_VALUE)
              : (r.nextInt(2001) - 1000) * 1000L + r.nextInt(1000);
      f.add("LongField l " + v + " " + yn(r));
    }
    if (r.nextInt(10) < 9) {
      f.add("FloatField f " + (r.nextInt(801) - 400) / 4f + " " + yn(r));
      if (r.nextInt(20) == 0) f.add("FloatField f " + (r.nextInt(801) - 400) / 4f + " N");
    }
    if (r.nextInt(10) < 9) f.add("DoubleField d " + (r.nextInt(1601) - 800) / 8.0 + " " + yn(r));
    if (r.nextInt(10) < 8) f.add("IntPoint ip1 " + (r.nextInt(41) - 20));
    if (r.nextInt(10) < 8)
      f.add("IntPoint ip2 " + (r.nextInt(41) - 20) + "," + (r.nextInt(41) - 20));
    if (r.nextInt(10) < 7) f.add("LongPoint lp " + (r.nextInt(201) - 100));
    if (r.nextInt(10) < 7) {
      float v = r.nextInt(12) == 0 ? (r.nextBoolean() ? -0f : 0f) : (r.nextInt(81) - 40) / 2f;
      f.add("FloatPoint fp " + v);
    }
    if (r.nextInt(10) < 7) f.add("DoublePoint dp " + (r.nextInt(81) - 40) / 4.0);
    if (r.nextInt(10) < 7)
      f.add("DoublePoint dp2 " + (r.nextInt(41) - 20) / 4.0 + "," + (r.nextInt(41) - 20) / 4.0);
    if (r.nextInt(10) < 6) f.add("BinaryPoint bp " + randomHex(r, 3) + "," + randomHex(r, 3));
    if (r.nextInt(10) < 6) f.add("BinaryPoint bp1 " + randomHex(r, 1) + "000000");
    if (r.nextInt(10) < 8) f.add("InetAddressPoint addr " + addr(r));
    if (r.nextInt(10) < 7) {
      int a = r.nextInt(201) - 100, b = r.nextInt(201) - 100;
      String min = a + "," + b, max = (a + r.nextInt(40)) + "," + (b + r.nextInt(40));
      f.add("IntRange ir " + min + " " + max);
      if (r.nextInt(10) < 6) f.add("IntRangeDocValuesField irdv " + min + " " + max);
    }
    if (r.nextInt(10) < 7) {
      long a = r.nextInt(2001) - 1000;
      String min = Long.toString(a), max = Long.toString(a + r.nextInt(300));
      f.add("LongRange lr " + min + " " + max);
      if (r.nextInt(10) < 6) f.add("LongRangeDocValuesField lrdv " + min + " " + max);
    }
    if (r.nextInt(10) < 6) {
      float a = (r.nextInt(401) - 200) / 4f;
      String min = Float.toString(a), max = Float.toString(a + r.nextInt(80) / 4f);
      f.add("FloatRange fr " + min + " " + max);
      if (r.nextInt(10) < 6) f.add("FloatRangeDocValuesField frdv " + min + " " + max);
    }
    if (r.nextInt(10) < 6) {
      double a = (r.nextInt(401) - 200) / 8.0, b = (r.nextInt(401) - 200) / 8.0;
      String min = a + "," + b, max = (a + r.nextInt(80) / 8.0) + "," + (b + r.nextInt(80) / 8.0);
      f.add("DoubleRange drg " + min + " " + max);
      if (r.nextInt(10) < 6) f.add("DoubleRangeDocValuesField drdv " + min + " " + max);
    }
    if (r.nextInt(10) < 5) {
      int x = r.nextInt(4), y = r.nextInt(200);
      f.add("InetAddressRange ipr 10.0." + x + "." + y + " 10.0." + (x + r.nextInt(2)) + "." + (y + r.nextInt(50)));
    }
    for (int i = 0, n = r.nextInt(3); i < n; i++) f.add("KeywordField kw k" + r.nextInt(12) + " " + yn(r));
    if (r.nextInt(10) < 8) f.add("SortedDocValuesField sdv w" + r.nextInt(15) + " 0");
    if (r.nextInt(10) < 8) f.add("SortedDocValuesField sdvi w" + r.nextInt(15) + " 1");
    for (int i = 0, n = r.nextInt(4); i < n; i++) f.add("SortedSetDocValuesField ssdv s" + r.nextInt(10) + " 0");
    for (int i = 0, n = r.nextInt(3); i < n; i++) f.add("SortedSetDocValuesField ssdvi s" + r.nextInt(10) + " 1");
    if (r.nextInt(100) < 85) f.add("NumericDocValuesField ndv " + (r.nextInt(1001) - 500) + " 0");
    if (r.nextInt(100) < 85) f.add("NumericDocValuesField ndvi " + (r.nextInt(1001) - 500) + " 1");
    for (int i = 0, n = r.nextInt(4); i < n; i++) f.add("SortedNumericDocValuesField sndv " + (r.nextInt(101) - 50) + " 0");
    for (int i = 0, n = r.nextInt(3); i < n; i++) f.add("SortedNumericDocValuesField sndvi " + (r.nextInt(101) - 50) + " 1");
    if (r.nextInt(10) < 4) f.add("BinaryDocValuesField bdv " + randomHex(r, 1 + r.nextInt(8)));
    if (r.nextInt(10) < 7) f.add("FeatureField feat pagerank " + (r.nextInt(100000) / 100f + 0.01f));
    if (r.nextInt(10) < 5) f.add("FeatureField feat popularity " + (r.nextInt(5000) / 10f + 0.1f));
    if (r.nextInt(10) < 3) f.add("FeatureField feat freshness " + (r.nextInt(5000) / 1000f + 0.001f));
    if (r.nextInt(10) < 3) {
      StringBuilder sb = new StringBuilder();
      for (int v = 0, n = 1 + r.nextInt(3); v < n; v++) {
        if (v > 0) sb.append(';');
        sb.append((r.nextInt(21) - 10) / 2f).append(',').append((r.nextInt(21) - 10) / 2f)
            .append(',').append((r.nextInt(21) - 10) / 2f);
      }
      f.add("LateInteractionField li " + sb);
    }
    return f;
  }

  // --- facets ------------------------------------------------------------------------------

  static String typeString(IndexableField f) {
    var t = f.fieldType();
    return (t.stored() ? 1 : 0) + "," + (t.tokenized() ? 1 : 0) + "," + t.indexOptions() + ","
        + (t.omitNorms() ? 1 : 0) + "," + t.docValuesType() + "," + t.docValuesSkipIndexType()
        + "," + t.pointDimensionCount() + "," + t.pointIndexDimensionCount() + ","
        + t.pointNumBytes() + "," + t.vectorDimension();
  }

  static String numeric(IndexableField f) {
    Number n;
    try {
      n = f.numericValue();
    } catch (IllegalStateException e) {
      return "-";
    }
    if (n == null) return "-";
    if (n instanceof Integer i) return "I:" + i;
    if (n instanceof Long l) return "L:" + l;
    if (n instanceof Float x) return "F:" + Integer.toHexString(Float.floatToRawIntBits(x));
    if (n instanceof Double x) return "D:" + Long.toHexString(Double.doubleToRawLongBits(x));
    throw new IllegalStateException(n.getClass().toString());
  }

  static String stored(IndexableField f) {
    if (f.fieldType().stored() == false) return "-";
    StoredValue v = f.storedValue();
    return switch (v.getType()) {
      case INTEGER -> "int:" + v.getIntValue();
      case LONG -> "long:" + v.getLongValue();
      case FLOAT -> "float:" + Integer.toHexString(Float.floatToRawIntBits(v.getFloatValue()));
      case DOUBLE -> "double:" + Long.toHexString(Double.doubleToRawLongBits(v.getDoubleValue()));
      case STRING -> "string:" + HEX.formatHex(v.getStringValue().getBytes(StandardCharsets.UTF_8));
      case BINARY -> "binary:" + hex(v.getBinaryValue());
      default -> throw new IllegalStateException();
    };
  }

  static String tokens(IndexableField f) throws IOException {
    if (f.fieldType().indexOptions() == IndexOptions.NONE) return "-";
    if (f.invertableType() == org.apache.lucene.document.InvertableType.BINARY) return "-";
    StringBuilder sb = new StringBuilder();
    try (TokenStream ts = f.tokenStream(ANALYZER, null)) {
      TermToBytesRefAttribute term = ts.addAttribute(TermToBytesRefAttribute.class);
      PositionIncrementAttribute pos = ts.addAttribute(PositionIncrementAttribute.class);
      OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);
      TermFrequencyAttribute freq = ts.addAttribute(TermFrequencyAttribute.class);
      ts.reset();
      boolean first = true;
      while (ts.incrementToken()) {
        if (!first) sb.append(';');
        first = false;
        sb.append(hex(term.getBytesRef())).append('/').append(pos.getPositionIncrement())
            .append('/').append(off.startOffset()).append('/').append(off.endOffset())
            .append('/').append(freq.getTermFrequency());
      }
      ts.end();
      sb.append('|').append(pos.getPositionIncrement()).append('/').append(off.endOffset());
    }
    return sb.toString();
  }

  static String facets(IndexableField f) throws IOException {
    BytesRef b = f.binaryValue();
    String inv =
        f.fieldType().indexOptions() == IndexOptions.NONE ? "-" : f.invertableType().toString();
    return f.name() + "\t" + typeString(f) + "\t" + (b == null ? "-" : hex(b)) + "\t"
        + numeric(f) + "\t" + stored(f) + "\t" + inv + "\t" + tokens(f);
  }

  // --- queries -----------------------------------------------------------------------------

  static final String[] QUERIES = {
    "IntPoint.newRangeQuery ip1 -5 10",
    "IntPoint.newExactQuery ip1 3",
    "IntPoint.newSetQuery ip1 1,2,3,-7,20",
    "IntPoint.newRangeQueryND ip2 -5,-5 5,10",
    "IntPoint.newRangeQuery ip1 10 -5",
    "IntPoint.newRangeQuery nofield 0 1",
    "LongPoint.newRangeQuery lp -20 30",
    "LongPoint.newExactQuery lp 7",
    "LongPoint.newSetQuery lp 1,5,9,99,-3",
    "FloatPoint.newRangeQuery fp -3.5 4.0",
    "FloatPoint.newRangeQueryExclusive fp -3.5 4.0",
    "FloatPoint.newExactQuery fp 0.0",
    "FloatPoint.newExactQuery fp -0.0",
    "FloatPoint.newSetQuery fp 1.5,-2.0,0.0,19.5",
    "DoublePoint.newRangeQuery dp -2.25 5.5",
    "DoublePoint.newRangeQueryExclusive dp -2.25 5.5",
    "DoublePoint.newExactQuery dp 1.25",
    "DoublePoint.newSetQuery dp 1.25,-3.0,9.75",
    "DoublePoint.newRangeQueryND dp2 -1.0,-2.0 2.0,3.0",
    "BinaryPoint.newRangeQueryND bp 200000,000000 ffffff,800000",
    "BinaryPoint.newRangeQuery bp1 40000000 bf000000",
    "BinaryPoint.newExactQuery bp1 7f000000",
    "BinaryPoint.newSetQuery bp1 01000000,7f000000,ff000000,80000000",
    "InetAddressPoint.newExactQuery addr 10.0.1.5",
    "InetAddressPoint.newPrefixQuery addr 10.0.1.0 24",
    "InetAddressPoint.newPrefixQuery addr 10.0.0.0 8",
    "InetAddressPoint.newPrefixQuery addr 2001:db8:: 32",
    "InetAddressPoint.newPrefixQuery addr 2001:db8::1:0 112",
    "InetAddressPoint.newRangeQuery addr 10.0.0.100 10.0.2.10",
    "InetAddressPoint.newSetQuery addr 10.0.0.1,10.0.1.2,10.0.3.255,2001:db8::1:1",
    "IntField.newRangeQuery i -10 10",
    "IntField.newExactQuery i 5",
    "IntField.newSetQuery i 1,2,3,50,-50",
    "LongField.newRangeQuery l -100000 250000",
    "LongField.newExactQuery l -9223372036854775808",
    "LongField.newSetQuery l -9223372036854775808,9223372036854775807,5",
    "FloatField.newRangeQuery f -10.5 20.25",
    "FloatField.newExactQuery f -87.75",
    "FloatField.newSetQuery f 1.0,-2.25,99.75,0.0",
    "DoubleField.newRangeQuery d -3.125 50.0",
    "DoubleField.newExactQuery d -75.5",
    "DoubleField.newSetQuery d 1.0,2.125,-0.5,99.875",
    "IntRange.newIntersectsQuery ir -10,-10 10,10",
    "IntRange.newWithinQuery ir -60,-60 60,60",
    "IntRange.newContainsQuery ir 0,0 1,1",
    "IntRange.newCrossesQuery ir -30,-30 30,30",
    "LongRange.newIntersectsQuery lr -100 100",
    "LongRange.newWithinQuery lr -500 500",
    "LongRange.newContainsQuery lr 10 20",
    "LongRange.newCrossesQuery lr -300 300",
    "FloatRange.newIntersectsQuery fr -5.5 5.5",
    "FloatRange.newWithinQuery fr -30.0 30.0",
    "FloatRange.newContainsQuery fr 1.0 1.25",
    "FloatRange.newCrossesQuery fr -20.0 20.0",
    "DoubleRange.newIntersectsQuery drg -3.0,-3.0 3.0,3.0",
    "DoubleRange.newWithinQuery drg -15.0,-15.0 15.0,15.0",
    "DoubleRange.newContainsQuery drg 0.0,0.0 0.5,0.5",
    "DoubleRange.newCrossesQuery drg -10.0,-10.0 10.0,10.0",
    "InetAddressRange.newIntersectsQuery ipr 10.0.1.0 10.0.1.100",
    "InetAddressRange.newWithinQuery ipr 10.0.0.0 10.0.2.255",
    "InetAddressRange.newContainsQuery ipr 10.0.2.100 10.0.2.101",
    "InetAddressRange.newCrossesQuery ipr 10.0.1.0 10.0.2.0",
    "IntRangeDocValuesField.newSlowIntersectsQuery irdv -10,-10 10,10",
    "LongRangeDocValuesField.newSlowIntersectsQuery lrdv -100 100",
    "FloatRangeDocValuesField.newSlowIntersectsQuery frdv -5.5 5.5",
    "DoubleRangeDocValuesField.newSlowIntersectsQuery drdv -3.0,-3.0 3.0,3.0",
    "NumericDocValuesField.newSlowRangeQuery ndv -100 100",
    "NumericDocValuesField.newSlowExactQuery ndv 460",
    "NumericDocValuesField.newSlowSetQuery ndv 1,2,3,-500,500,250",
    "NumericDocValuesField.newSlowRangeQuery ndv -9223372036854775808 9223372036854775807",
    "NumericDocValuesField.newSlowRangeQuery ndv 5 -5",
    "NumericDocValuesField.newSlowRangeQuery ndvi -100 100",
    "NumericDocValuesField.newSlowRangeQuery ndvi -1000 1000",
    "NumericDocValuesField.newSlowRangeQuery ndvi 600 900",
    "NumericDocValuesField.newSlowSetQuery ndvi 1,2,3,-500,500,250",
    "SortedNumericDocValuesField.newSlowRangeQuery sndv -5 5",
    "SortedNumericDocValuesField.newSlowExactQuery sndv 0",
    "SortedNumericDocValuesField.newSlowSetQuery sndv -50,50,0,7",
    "SortedNumericDocValuesField.newSlowRangeQuery sndvi -5 5",
    "SortedNumericDocValuesField.newSlowSetQuery sndvi -50,50,0,7",
    "SortedDocValuesField.newSlowRangeQuery sdv w3 w9 true false",
    "SortedDocValuesField.newSlowRangeQuery sdv * w2 true true",
    "SortedDocValuesField.newSlowRangeQuery sdv w5 * false true",
    "SortedDocValuesField.newSlowRangeQuery sdv * * true true",
    "SortedDocValuesField.newSlowRangeQuery sdv w10a w12 true true",
    "SortedDocValuesField.newSlowExactQuery sdv w4",
    "SortedDocValuesField.newSlowSetQuery sdv w1,w2,wx",
    "SortedDocValuesField.newSlowRangeQuery sdvi w3 w9 true false",
    "SortedDocValuesField.newSlowSetQuery sdvi w1,w14,w7",
    "SortedSetDocValuesField.newSlowRangeQuery ssdv s2 s5 false true",
    "SortedSetDocValuesField.newSlowExactQuery ssdv s7",
    "SortedSetDocValuesField.newSlowSetQuery ssdv s1,s9,zz",
    "SortedSetDocValuesField.newSlowRangeQuery ssdvi s2 s5 true true",
    "SortedSetDocValuesField.newSlowSetQuery ssdvi s0,s3",
    "KeywordField.newExactQuery kw k3",
    "KeywordField.newExactQuery kw nope",
    "KeywordField.newSetQuery kw k1,k2,k9",
    "FeatureField.newLinearQuery feat pagerank 1.0",
    "FeatureField.newLinearQuery feat popularity 2.5",
    "FeatureField.newLogQuery feat pagerank 1.0 4.5",
    "FeatureField.newLogQuery feat popularity 3.0 1.0",
    "FeatureField.newSaturationQuery feat popularity 1.0 10.0",
    "FeatureField.newSaturationQuery feat pagerank 7.0 0.5",
    "FeatureField.newSaturationQueryAuto feat pagerank",
    "FeatureField.newSaturationQueryAuto feat freshness",
    "FeatureField.newSaturationQueryAuto feat missing",
    "FeatureField.newSigmoidQuery feat freshness 2.0 0.5 0.7",
    "FeatureField.newSigmoidQuery feat pagerank 1.0 100.0 2.0",
    "FeatureField.newLinearQuery feat missing 1.0",
    "LongField.newDistanceFeatureQuery l 2.0 1000 50000",
    "LongField.newDistanceFeatureQuery l 1.0 0 1",
    "LongField.newDistanceFeatureQuery l 1.0 9223372036854775807 1000000",
  };

  static Query query(String spec) {
    String[] a = spec.split(" ");
    String f = a[1];
    switch (a[0]) {
      case "IntPoint.newRangeQuery":
        return IntPoint.newRangeQuery(f, Integer.parseInt(a[2]), Integer.parseInt(a[3]));
      case "IntPoint.newExactQuery":
        return IntPoint.newExactQuery(f, Integer.parseInt(a[2]));
      case "IntPoint.newSetQuery":
        return IntPoint.newSetQuery(f, ints(a[2]));
      case "IntPoint.newRangeQueryND":
        return IntPoint.newRangeQuery(f, ints(a[2]), ints(a[3]));
      case "LongPoint.newRangeQuery":
        return LongPoint.newRangeQuery(f, Long.parseLong(a[2]), Long.parseLong(a[3]));
      case "LongPoint.newExactQuery":
        return LongPoint.newExactQuery(f, Long.parseLong(a[2]));
      case "LongPoint.newSetQuery":
        return LongPoint.newSetQuery(f, longs(a[2]));
      case "FloatPoint.newRangeQuery":
        return FloatPoint.newRangeQuery(f, Float.parseFloat(a[2]), Float.parseFloat(a[3]));
      case "FloatPoint.newRangeQueryExclusive":
        return FloatPoint.newRangeQuery(
            f,
            FloatPoint.nextUp(Float.parseFloat(a[2])),
            FloatPoint.nextDown(Float.parseFloat(a[3])));
      case "FloatPoint.newExactQuery":
        return FloatPoint.newExactQuery(f, Float.parseFloat(a[2]));
      case "FloatPoint.newSetQuery":
        return FloatPoint.newSetQuery(f, floats(a[2]));
      case "DoublePoint.newRangeQuery":
        return DoublePoint.newRangeQuery(f, Double.parseDouble(a[2]), Double.parseDouble(a[3]));
      case "DoublePoint.newRangeQueryExclusive":
        return DoublePoint.newRangeQuery(
            f,
            DoublePoint.nextUp(Double.parseDouble(a[2])),
            DoublePoint.nextDown(Double.parseDouble(a[3])));
      case "DoublePoint.newExactQuery":
        return DoublePoint.newExactQuery(f, Double.parseDouble(a[2]));
      case "DoublePoint.newSetQuery":
        return DoublePoint.newSetQuery(f, doubles(a[2]));
      case "DoublePoint.newRangeQueryND":
        return DoublePoint.newRangeQuery(f, doubles(a[2]), doubles(a[3]));
      case "BinaryPoint.newRangeQueryND":
        return BinaryPoint.newRangeQuery(f, byteDims(a[2]), byteDims(a[3]));
      case "BinaryPoint.newRangeQuery":
        return BinaryPoint.newRangeQuery(f, unhex(a[2]), unhex(a[3]));
      case "BinaryPoint.newExactQuery":
        return BinaryPoint.newExactQuery(f, unhex(a[2]));
      case "BinaryPoint.newSetQuery":
        return BinaryPoint.newSetQuery(f, byteDims(a[2]));
      case "InetAddressPoint.newExactQuery":
        return InetAddressPoint.newExactQuery(f, ip(a[2]));
      case "InetAddressPoint.newPrefixQuery":
        return InetAddressPoint.newPrefixQuery(f, ip(a[2]), Integer.parseInt(a[3]));
      case "InetAddressPoint.newRangeQuery":
        return InetAddressPoint.newRangeQuery(f, ip(a[2]), ip(a[3]));
      case "InetAddressPoint.newSetQuery":
        return InetAddressPoint.newSetQuery(
            f, Arrays.stream(a[2].split(",")).map(GenDocumentFields::ip).toArray(InetAddress[]::new));
      case "IntField.newRangeQuery":
        return IntField.newRangeQuery(f, Integer.parseInt(a[2]), Integer.parseInt(a[3]));
      case "IntField.newExactQuery":
        return IntField.newExactQuery(f, Integer.parseInt(a[2]));
      case "IntField.newSetQuery":
        return IntField.newSetQuery(f, ints(a[2]));
      case "LongField.newRangeQuery":
        return LongField.newRangeQuery(f, Long.parseLong(a[2]), Long.parseLong(a[3]));
      case "LongField.newExactQuery":
        return LongField.newExactQuery(f, Long.parseLong(a[2]));
      case "LongField.newSetQuery":
        return LongField.newSetQuery(f, longs(a[2]));
      case "FloatField.newRangeQuery":
        return FloatField.newRangeQuery(f, Float.parseFloat(a[2]), Float.parseFloat(a[3]));
      case "FloatField.newExactQuery":
        return FloatField.newExactQuery(f, Float.parseFloat(a[2]));
      case "FloatField.newSetQuery":
        return FloatField.newSetQuery(f, floats(a[2]));
      case "DoubleField.newRangeQuery":
        return DoubleField.newRangeQuery(f, Double.parseDouble(a[2]), Double.parseDouble(a[3]));
      case "DoubleField.newExactQuery":
        return DoubleField.newExactQuery(f, Double.parseDouble(a[2]));
      case "DoubleField.newSetQuery":
        return DoubleField.newSetQuery(f, doubles(a[2]));
      case "IntRange.newIntersectsQuery":
        return IntRange.newIntersectsQuery(f, ints(a[2]), ints(a[3]));
      case "IntRange.newWithinQuery":
        return IntRange.newWithinQuery(f, ints(a[2]), ints(a[3]));
      case "IntRange.newContainsQuery":
        return IntRange.newContainsQuery(f, ints(a[2]), ints(a[3]));
      case "IntRange.newCrossesQuery":
        return IntRange.newCrossesQuery(f, ints(a[2]), ints(a[3]));
      case "LongRange.newIntersectsQuery":
        return LongRange.newIntersectsQuery(f, longs(a[2]), longs(a[3]));
      case "LongRange.newWithinQuery":
        return LongRange.newWithinQuery(f, longs(a[2]), longs(a[3]));
      case "LongRange.newContainsQuery":
        return LongRange.newContainsQuery(f, longs(a[2]), longs(a[3]));
      case "LongRange.newCrossesQuery":
        return LongRange.newCrossesQuery(f, longs(a[2]), longs(a[3]));
      case "FloatRange.newIntersectsQuery":
        return FloatRange.newIntersectsQuery(f, floats(a[2]), floats(a[3]));
      case "FloatRange.newWithinQuery":
        return FloatRange.newWithinQuery(f, floats(a[2]), floats(a[3]));
      case "FloatRange.newContainsQuery":
        return FloatRange.newContainsQuery(f, floats(a[2]), floats(a[3]));
      case "FloatRange.newCrossesQuery":
        return FloatRange.newCrossesQuery(f, floats(a[2]), floats(a[3]));
      case "DoubleRange.newIntersectsQuery":
        return DoubleRange.newIntersectsQuery(f, doubles(a[2]), doubles(a[3]));
      case "DoubleRange.newWithinQuery":
        return DoubleRange.newWithinQuery(f, doubles(a[2]), doubles(a[3]));
      case "DoubleRange.newContainsQuery":
        return DoubleRange.newContainsQuery(f, doubles(a[2]), doubles(a[3]));
      case "DoubleRange.newCrossesQuery":
        return DoubleRange.newCrossesQuery(f, doubles(a[2]), doubles(a[3]));
      case "InetAddressRange.newIntersectsQuery":
        return InetAddressRange.newIntersectsQuery(f, ip(a[2]), ip(a[3]));
      case "InetAddressRange.newWithinQuery":
        return InetAddressRange.newWithinQuery(f, ip(a[2]), ip(a[3]));
      case "InetAddressRange.newContainsQuery":
        return InetAddressRange.newContainsQuery(f, ip(a[2]), ip(a[3]));
      case "InetAddressRange.newCrossesQuery":
        return InetAddressRange.newCrossesQuery(f, ip(a[2]), ip(a[3]));
      case "IntRangeDocValuesField.newSlowIntersectsQuery":
        return IntRangeDocValuesField.newSlowIntersectsQuery(f, ints(a[2]), ints(a[3]));
      case "LongRangeDocValuesField.newSlowIntersectsQuery":
        return LongRangeDocValuesField.newSlowIntersectsQuery(f, longs(a[2]), longs(a[3]));
      case "FloatRangeDocValuesField.newSlowIntersectsQuery":
        return FloatRangeDocValuesField.newSlowIntersectsQuery(f, floats(a[2]), floats(a[3]));
      case "DoubleRangeDocValuesField.newSlowIntersectsQuery":
        return DoubleRangeDocValuesField.newSlowIntersectsQuery(f, doubles(a[2]), doubles(a[3]));
      case "NumericDocValuesField.newSlowRangeQuery":
        return NumericDocValuesField.newSlowRangeQuery(f, Long.parseLong(a[2]), Long.parseLong(a[3]));
      case "NumericDocValuesField.newSlowExactQuery":
        return NumericDocValuesField.newSlowExactQuery(f, Long.parseLong(a[2]));
      case "NumericDocValuesField.newSlowSetQuery":
        return NumericDocValuesField.newSlowSetQuery(f, longs(a[2]));
      case "SortedNumericDocValuesField.newSlowRangeQuery":
        return SortedNumericDocValuesField.newSlowRangeQuery(
            f, Long.parseLong(a[2]), Long.parseLong(a[3]));
      case "SortedNumericDocValuesField.newSlowExactQuery":
        return SortedNumericDocValuesField.newSlowExactQuery(f, Long.parseLong(a[2]));
      case "SortedNumericDocValuesField.newSlowSetQuery":
        return SortedNumericDocValuesField.newSlowSetQuery(f, longs(a[2]));
      case "SortedDocValuesField.newSlowRangeQuery":
        return SortedDocValuesField.newSlowRangeQuery(
            f, bound(a[2]), bound(a[3]), Boolean.parseBoolean(a[4]), Boolean.parseBoolean(a[5]));
      case "SortedDocValuesField.newSlowExactQuery":
        return SortedDocValuesField.newSlowExactQuery(f, new BytesRef(a[2]));
      case "SortedDocValuesField.newSlowSetQuery":
        return SortedDocValuesField.newSlowSetQuery(f, byteRefs(a[2]));
      case "SortedSetDocValuesField.newSlowRangeQuery":
        return SortedSetDocValuesField.newSlowRangeQuery(
            f, bound(a[2]), bound(a[3]), Boolean.parseBoolean(a[4]), Boolean.parseBoolean(a[5]));
      case "SortedSetDocValuesField.newSlowExactQuery":
        return SortedSetDocValuesField.newSlowExactQuery(f, new BytesRef(a[2]));
      case "SortedSetDocValuesField.newSlowSetQuery":
        return SortedSetDocValuesField.newSlowSetQuery(f, byteRefs(a[2]));
      case "KeywordField.newExactQuery":
        return KeywordField.newExactQuery(f, a[2]);
      case "KeywordField.newSetQuery":
        return KeywordField.newSetQuery(f, byteRefs(a[2]));
      case "FeatureField.newLinearQuery":
        return FeatureField.newLinearQuery(f, a[2], Float.parseFloat(a[3]));
      case "FeatureField.newLogQuery":
        return FeatureField.newLogQuery(f, a[2], Float.parseFloat(a[3]), Float.parseFloat(a[4]));
      case "FeatureField.newSaturationQuery":
        return FeatureField.newSaturationQuery(
            f, a[2], Float.parseFloat(a[3]), Float.parseFloat(a[4]));
      case "FeatureField.newSaturationQueryAuto":
        return FeatureField.newSaturationQuery(f, a[2]);
      case "FeatureField.newSigmoidQuery":
        return FeatureField.newSigmoidQuery(
            f, a[2], Float.parseFloat(a[3]), Float.parseFloat(a[4]), Float.parseFloat(a[5]));
      case "LongField.newDistanceFeatureQuery":
        return LongField.newDistanceFeatureQuery(
            f, Float.parseFloat(a[2]), Long.parseLong(a[3]), Long.parseLong(a[4]));
      default:
        throw new IllegalArgumentException(spec);
    }
  }

  static BytesRef bound(String s) {
    return s.equals("*") ? null : new BytesRef(s);
  }

  /**
   * Every hit: {@code total}, then either {@code C:<scoreBits>} and the matching docs as ascending
   * runs {@code a-b} when every hit scored the same, or {@code S} and each hit as {@code
   * doc:scoreBits} in rank order (score descending, doc ascending).
   */
  static String hits(IndexSearcher searcher, Query q) throws IOException {
    TopDocs td = searcher.search(q, 100_000);
    StringBuilder sb = new StringBuilder();
    sb.append(td.totalHits.value());
    boolean constant = true;
    for (ScoreDoc sd : td.scoreDocs) {
      constant &= sd.score == td.scoreDocs[0].score;
    }
    if (constant) {
      int[] docs = Arrays.stream(td.scoreDocs).mapToInt(sd -> sd.doc).sorted().toArray();
      sb.append("\tC:")
          .append(docs.length == 0 ? "0" : Integer.toHexString(Float.floatToIntBits(td.scoreDocs[0].score)));
      for (int i = 0; i < docs.length; ) {
        int j = i;
        while (j + 1 < docs.length && docs[j + 1] == docs[j] + 1) j++;
        sb.append('\t').append(docs[i]).append('-').append(docs[j]);
        i = j + 1;
      }
    } else {
      sb.append("\tS");
      for (ScoreDoc sd : td.scoreDocs) {
        sb.append('\t').append(sd.doc).append(':').append(Integer.toHexString(Float.floatToIntBits(sd.score)));
      }
    }
    return sb.toString();
  }

  // --- sorts -------------------------------------------------------------------------------

  static final String[] SORTS = {
    "IntField i false MIN",
    "IntField i true MAX",
    "IntField i false MAX 100",
    "LongField l false MIN",
    "LongField l true MIN -5",
    "FloatField f true MIN",
    "FloatField f false MAX -1.5",
    "DoubleField d false MAX",
    "DoubleField d true MIN 0.25",
    "KeywordField kw false MIN",
    "KeywordField kw true MAX",
    "FeatureField feat pagerank",
    "FeatureField feat freshness",
  };

  static SortField sortField(String spec) {
    String[] a = spec.split(" ");
    String f = a[1];
    if (a[0].equals("FeatureField")) return FeatureField.newFeatureSort(f, a[2]);
    boolean reverse = Boolean.parseBoolean(a[2]);
    if (a[0].equals("KeywordField")) {
      return KeywordField.newSortField(f, reverse, SortedSetSelector.Type.valueOf(a[3]));
    }
    SortedNumericSelector.Type sel = SortedNumericSelector.Type.valueOf(a[3]);
    boolean missing = a.length > 4;
    return switch (a[0]) {
      case "IntField" -> missing
          ? IntField.newSortField(f, reverse, sel, Integer.parseInt(a[4]))
          : IntField.newSortField(f, reverse, sel);
      case "LongField" -> missing
          ? LongField.newSortField(f, reverse, sel, Long.parseLong(a[4]))
          : LongField.newSortField(f, reverse, sel);
      case "FloatField" -> missing
          ? FloatField.newSortField(f, reverse, sel, Float.parseFloat(a[4]))
          : FloatField.newSortField(f, reverse, sel);
      case "DoubleField" -> missing
          ? DoubleField.newSortField(f, reverse, sel, Double.parseDouble(a[4]))
          : DoubleField.newSortField(f, reverse, sel);
      default -> throw new IllegalArgumentException(spec);
    };
  }

  static String sortValue(Object o) {
    if (o == null) return "null";
    if (o instanceof Integer i) return "I:" + i;
    if (o instanceof Long l) return "L:" + l;
    if (o instanceof Float x) return "F:" + Integer.toHexString(Float.floatToIntBits(x));
    if (o instanceof Double x) return "D:" + Long.toHexString(Double.doubleToLongBits(x));
    if (o instanceof BytesRef b) return "B:" + hex(b);
    throw new IllegalStateException(o.getClass().toString());
  }

  // --- the index ---------------------------------------------------------------------------

  static void clean(Path out) throws IOException {
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("document_fields");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);

    Random r = new Random(20260930L);
    List<List<String>> docs = new ArrayList<>();
    StringBuilder docsOut = new StringBuilder();
    StringBuilder facetsOut = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(ANALYZER);
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
            int id = seg * DOCS_PER_SEGMENT + i;
            List<String> specs = docSpecs(r, id);
            docs.add(specs);
            docsOut.append(id);
            Document doc = new Document();
            int fi = 0;
            for (String s : specs) {
              docsOut.append('\t').append(s);
              IndexableField f = field(s);
              if (id % 5 == 0) facetsOut.append(id).append('\t').append(fi++).append('\t').append(facets(f)).append('\n');
              doc.add(f);
            }
            docsOut.append('\n');
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int i = 0; i < DOCS_PER_SEGMENT; i += 17) {
          w.deleteDocuments(new Term("id", Integer.toString(i)));
          w.deleteDocuments(new Term("id", Integer.toString(2 * DOCS_PER_SEGMENT + i)));
        }
        w.commit();
      }
      Files.writeString(root.resolve("docs.tsv"), docsOut);
      Files.writeString(root.resolve("facets.tsv"), facetsOut);

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != SEGMENTS) throw new AssertionError("segments");
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        StringBuilder q = new StringBuilder();
        for (String spec : QUERIES) {
          q.append(spec).append('\t').append(hits(searcher, query(spec))).append('\n');
        }
        Files.writeString(root.resolve("queries.tsv"), q);

        StringBuilder s = new StringBuilder();
        for (String spec : SORTS) {
          TopFieldDocs td = searcher.search(new MatchAllDocsQuery(), 40, new Sort(sortField(spec)));
          s.append(spec).append('\t').append(td.totalHits.value());
          for (ScoreDoc sd : td.scoreDocs) {
            s.append('\t').append(sd.doc).append(':').append(sortValue(((FieldDoc) sd).fields[0]));
          }
          s.append('\n');
        }
        Files.writeString(root.resolve("sorts.tsv"), s);

        StringBuilder v = new StringBuilder();
        for (String feature : new String[] {"pagerank", "popularity", "freshness", "missing"}) {
          DoubleValuesSource src = FeatureField.newDoubleValues("feat", feature);
          for (LeafReaderContext leaf : reader.leaves()) {
            DoubleValues dv = src.getValues(leaf, null);
            for (int d = 0; d < leaf.reader().maxDoc(); d++) {
              if (dv.advanceExact(d)) {
                v.append(leaf.docBase + d).append('\t').append(feature).append('\t')
                    .append(Long.toHexString(Double.doubleToLongBits(dv.doubleValue()))).append('\n');
              }
            }
          }
        }
        Files.writeString(root.resolve("values.tsv"), v);
      }
    }

    Files.writeString(root.resolve("dates.tsv"), dates());
    sortedIndex(Path.of(args[0]).resolve("document_fields_sorted_num"), true);
    sortedIndex(Path.of(args[0]).resolve("document_fields_sorted_kw"), false);
  }

  static String dates() {
    StringBuilder sb = new StringBuilder();
    long[] times = {
      0L, -1L, 1L, 1_095_774_611_123L, 951_782_400_000L, 4_102_444_799_999L,
      -2_208_988_800_000L, -12_219_292_800_000L, -12_219_292_800_001L, -30_610_224_000_000L,
      -62_135_596_800_000L, -62_198_755_200_000L, 253_402_300_799_999L, 253_402_300_800_000L,
      1_234_567_890_123L, -987_654_321_987L
    };
    for (long t : times) {
      for (DateTools.Resolution res : DateTools.Resolution.values()) {
        sb.append("t\t").append(t).append('\t').append(res).append('\t')
            .append(DateTools.timeToString(t, res)).append('\t').append(DateTools.round(t, res))
            .append('\n');
      }
    }
    String[] strings = {
      "2004", "200409", "20040921", "2004092113", "200409211350", "20040921135011",
      "20040921135011123", "200413", "20040100", "20041231245960", "15821004", "15821015",
      "15821010", "00010101", "10000101", "99991231235959999", "2004a", "20041", "", "abcd",
      "123456789012345678"
    };
    for (String s : strings) {
      String v;
      try {
        v = Long.toString(DateTools.stringToTime(s));
      } catch (java.text.ParseException e) {
        v = "ERR";
      }
      sb.append("s\t").append(s).append('\t').append(v).append('\n');
    }
    return sb.toString();
  }

  static final String[] SORTED_NUM_QUERIES = {
    "NumericDocValuesField.newSlowRangeQuery snum 100 200",
    "NumericDocValuesField.newSlowRangeQuery snum 0 0",
    "NumericDocValuesField.newSlowRangeQuery snum 2999 5000",
    "NumericDocValuesField.newSlowRangeQuery snum -5 50",
    "NumericDocValuesField.newSlowRangeQuery snum 1500 1500",
    "NumericDocValuesField.newSlowRangeQuery snum 3000 9000",
    "NumericDocValuesField.newSlowRangeQuery snum -9000 -1",
    "NumericDocValuesField.newSlowRangeQuery snum 1 2998",
    "NumericDocValuesField.newSlowExactQuery snum 777",
    "SortedDocValuesField.newSlowRangeQuery skw k0100 k0200 true false",
  };

  static final String[] SORTED_KW_QUERIES = {
    "SortedDocValuesField.newSlowRangeQuery skw k0100 k0200 true false",
    "SortedDocValuesField.newSlowRangeQuery skw k0100 k0200 false true",
    "SortedDocValuesField.newSlowRangeQuery skw * k0005 true true",
    "SortedDocValuesField.newSlowRangeQuery skw k2990 * true true",
    "SortedDocValuesField.newSlowRangeQuery skw k1234 k1234 true true",
    "SortedDocValuesField.newSlowRangeQuery skw k1234a k1235 true true",
    "SortedDocValuesField.newSlowRangeQuery skw a b true true",
    "SortedDocValuesField.newSlowRangeQuery skw zz zzz true true",
    "SortedDocValuesField.newSlowExactQuery skw k0042",
    "NumericDocValuesField.newSlowRangeQuery snum 100 200",
  };

  /** An index sorted by a skip-indexed field: the numeric one (descending) or the keyword one. */
  static void sortedIndex(Path root, boolean numeric) throws IOException {
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(numeric ? 11 : 12);
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(ANALYZER);
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      cfg.setIndexSort(
          new Sort(
              numeric
                  ? new SortField("snum", SortField.Type.LONG, true)
                  : new SortField("skw", SortField.Type.STRING, false)));
      int id = 0;
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < 2; seg++) {
          int n = seg == 0 ? 12_000 : 3_000;
          for (int i = 0; i < n; i++, id++) {
            List<String> specs = new ArrayList<>();
            specs.add("id " + id);
            specs.add("NumericDocValuesField snum " + r.nextInt(3000) + " 1");
            specs.add("SortedDocValuesField skw k" + String.format("%04d", r.nextInt(3000)) + " 1");
            Document doc = new Document();
            for (String s : specs) {
              doc.add(field(s));
            }
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int i = 0; i < id; i += 101) w.deleteDocuments(new Term("id", Integer.toString(i)));
        w.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        StringBuilder q = new StringBuilder();
        for (String spec : numeric ? SORTED_NUM_QUERIES : SORTED_KW_QUERIES) {
          q.append(spec).append('\t').append(hits(searcher, query(spec))).append('\n');
        }
        Files.writeString(root.resolve("queries.tsv"), q);
      }
    }
  }
}
