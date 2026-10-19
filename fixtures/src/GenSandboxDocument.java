import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.sandbox.document.BigIntegerPoint;
import org.apache.lucene.sandbox.document.DoublePointMultiRangeBuilder;
import org.apache.lucene.sandbox.document.FloatPointMultiRangeBuilder;
import org.apache.lucene.sandbox.document.FloatPointNearestNeighbor;
import org.apache.lucene.sandbox.document.HalfFloatPoint;
import org.apache.lucene.sandbox.document.IntPointMultiRangeBuilder;
import org.apache.lucene.sandbox.document.LatLonBoundingBox;
import org.apache.lucene.sandbox.document.LongPointMultiRangeBuilder;
import org.apache.lucene.sandbox.search.DocValuesMultiRangeQuery;
import org.apache.lucene.sandbox.search.MultiRangeQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * lucene-sandbox's document package and its multi-range queries, recorded from Lucene 10.5.0 for
 * crates/lucene-search/tests/sandbox_document_fixtures.rs. Written to {@code sandbox_document/}:
 *
 * <pre>
 *   index/         three segments of 200 documents, every 13th id deleted in segments 0 and 2
 *   docs.tsv       doc id, then one field spec per column
 *   packed.tsv     each sandbox field spec, then its binaryValue as hex
 *   half.tsv       HalfFloatPoint per float: float bits, sortable short, its float's bits,
 *                  nextUp bits, nextDown bits, encodeDimension hex
 *   big.tsv        BigIntegerPoint per value: the value, encodeDimension hex
 *   queries.tsv    query spec, then its hits (see hits())
 *   nearest.tsv    FloatPointNearestNeighbor.nearest spec, then total hits and doc:distance bits
 *   errors.tsv     spec, then the IllegalArgumentException message it throws
 *   large/         one segment of 9,000 documents, every 9th deleted (index/, docs.tsv), and
 *                  queries.tsv: spec, search(q, 10)'s total hits, relation and docs,
 *                  count(q), then hits()
 * </pre>
 */
public class GenSandboxDocument {
  static final int SEGMENTS = 3;
  static final int DOCS_PER_SEGMENT = 200;
  static final HexFormat HEX = HexFormat.of();

  static float[] floats(String s) {
    String[] p = s.split(",");
    float[] f = new float[p.length];
    for (int i = 0; i < p.length; i++) f[i] = Float.parseFloat(p[i]);
    return f;
  }

  static double[] doubles(String s) {
    return Arrays.stream(s.split(",")).mapToDouble(Double::parseDouble).toArray();
  }

  static int[] ints(String s) {
    return Arrays.stream(s.split(",")).mapToInt(Integer::parseInt).toArray();
  }

  static long[] longs(String s) {
    return Arrays.stream(s.split(",")).mapToLong(Long::parseLong).toArray();
  }

  static BigInteger[] bigs(String s) {
    return Arrays.stream(s.split(",")).map(BigInteger::new).toArray(BigInteger[]::new);
  }

  /** Builds one field from its spec. */
  static IndexableField field(String spec) {
    String[] a = spec.split(" ");
    String f = a.length > 1 ? a[1] : "";
    switch (a[0]) {
      case "id":
        return new StringField("id", a[1], Field.Store.NO);
      case "HalfFloatPoint":
        return new HalfFloatPoint(f, floats(a[2]));
      case "BigIntegerPoint":
        return new BigIntegerPoint(f, bigs(a[2]));
      case "LatLonBoundingBox":
        {
          double[] d = doubles(a[2]);
          return new LatLonBoundingBox(f, d[0], d[1], d[2], d[3]);
        }
      case "IntPoint":
        return new IntPoint(f, ints(a[2]));
      case "LongPoint":
        return new LongPoint(f, longs(a[2]));
      case "FloatPoint":
        return new FloatPoint(f, floats(a[2]));
      case "DoublePoint":
        return new DoublePoint(f, doubles(a[2]));
      case "NumericDocValuesField":
        return a[3].equals("1")
            ? NumericDocValuesField.indexedField(f, Long.parseLong(a[2]))
            : new NumericDocValuesField(f, Long.parseLong(a[2]));
      case "SortedNumericDocValuesField":
        return a[3].equals("1")
            ? SortedNumericDocValuesField.indexedField(f, Long.parseLong(a[2]))
            : new SortedNumericDocValuesField(f, Long.parseLong(a[2]));
      case "SortedDocValuesField":
        return a[3].equals("1")
            ? SortedDocValuesField.indexedField(f, new BytesRef(a[2]))
            : new SortedDocValuesField(f, new BytesRef(a[2]));
      case "SortedSetDocValuesField":
        return a[3].equals("1")
            ? SortedSetDocValuesField.indexedField(f, new BytesRef(a[2]))
            : new SortedSetDocValuesField(f, new BytesRef(a[2]));
      default:
        throw new IllegalArgumentException(spec);
    }
  }

  // --- documents ---------------------------------------------------------------------------

  static final float[] HALF_SPECIALS = {
    0f, -0f, 1f, -1f, 1.00048828125f, 1.00146484375f, 65504f, 65519.99f, 65520f, -65520f,
    6.1035156e-5f, 5.9604645e-8f, 2.9802322e-8f, 2.9802326e-8f, 1e-8f, -1e-8f, 0.1f, 1.5f, 2.5f,
    Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.MIN_VALUE, Float.MAX_VALUE, 1e-40f,
    3.14159f, -2.71828f, 1000.4f, 1000.6f, 2049f, 2051f, 4097f
  };

  static String halfValue(Random r) {
    if (r.nextInt(5) == 0) return Float.toString(HALF_SPECIALS[r.nextInt(HALF_SPECIALS.length)]);
    return Float.toString((r.nextInt(40001) - 20000) / 100f);
  }

  static String bigValue(Random r) {
    switch (r.nextInt(6)) {
      case 0:
        return new BigInteger(127, r).subtract(BigInteger.ONE.shiftLeft(126)).toString();
      case 1:
        return r.nextBoolean()
            ? BigIntegerPoint.MIN_VALUE.toString()
            : BigIntegerPoint.MAX_VALUE.toString();
      case 2:
        return BigInteger.valueOf(r.nextLong()).shiftLeft(40).toString();
      default:
        return Integer.toString(r.nextInt(201) - 100);
    }
  }

  static String bbox(Random r) {
    double lat1 = r.nextInt(1601) / 10.0 - 80;
    double lat2 = Math.min(90, lat1 + r.nextInt(300) / 10.0);
    double lon1 = r.nextInt(3401) / 10.0 - 170;
    double lon2 = Math.min(180, lon1 + r.nextInt(400) / 10.0);
    return lat1 + "," + lon1 + "," + lat2 + "," + lon2;
  }

  static String term(Random r) {
    return String.format("t%02d", r.nextInt(40));
  }

  static List<String> docSpecs(Random r, int id) {
    List<String> s = new ArrayList<>();
    s.add("id " + id);
    if (r.nextInt(8) != 0) s.add("HalfFloatPoint h " + halfValue(r));
    if (r.nextInt(4) == 0) s.add("HalfFloatPoint h " + halfValue(r));
    if (r.nextInt(3) != 0) s.add("HalfFloatPoint h2 " + halfValue(r) + "," + halfValue(r));
    if (r.nextInt(8) != 0) s.add("BigIntegerPoint b " + bigValue(r));
    if (r.nextInt(5) == 0) s.add("BigIntegerPoint b " + bigValue(r));
    if (r.nextInt(3) != 0) s.add("BigIntegerPoint b2 " + bigValue(r) + "," + bigValue(r));
    if (r.nextInt(6) != 0) s.add("LatLonBoundingBox bb " + bbox(r));
    if (r.nextInt(6) == 0) s.add("LatLonBoundingBox bb " + bbox(r));
    int nv = r.nextInt(4) == 0 ? 0 : 1 + r.nextInt(2);
    for (int i = 0; i < nv; i++) {
      s.add(
          "FloatPoint v "
              + (r.nextInt(2001) - 1000) / 100f
              + ","
              + (r.nextInt(2001) - 1000) / 100f
              + ","
              + (r.nextInt(201) - 100) / 10f);
    }
    if (r.nextInt(10) != 0) s.add("IntPoint i " + (r.nextInt(2001) - 1000));
    if (r.nextInt(4) == 0) s.add("IntPoint i " + (r.nextInt(2001) - 1000));
    if (r.nextInt(5) != 0) s.add("IntPoint i2 " + (r.nextInt(201) - 100) + "," + (r.nextInt(201) - 100));
    if (r.nextInt(10) != 0) s.add("LongPoint l " + ((long) (r.nextInt(2001) - 1000) << 33));
    if (r.nextInt(10) != 0) s.add("FloatPoint f " + (r.nextInt(2001) - 1000) / 8f);
    if (r.nextInt(10) != 0) s.add("DoublePoint d " + (r.nextInt(2001) - 1000) / 16.0);
    if (r.nextInt(5) != 0) {
      s.add(
          "DoublePoint d2 "
              + (r.nextInt(201) - 100) / 4.0
              + ","
              + (r.nextInt(201) - 100) / 4.0);
    }
    int nsn = r.nextInt(4);
    for (int i = 0; i < nsn; i++) {
      long v = r.nextInt(30) == 0 ? (r.nextBoolean() ? Long.MIN_VALUE : Long.MAX_VALUE) : r.nextInt(101) - 50;
      s.add("SortedNumericDocValuesField sn " + v + " 0");
    }
    // Singleton, skip-indexed, sparse: `matches()` re-reads a later document's one value.
    if (id % 3 != 0) s.add("NumericDocValuesField nx " + (id % 200) + " 1");
    // Skip-indexed, multi-valued: a stale read takes 5 then the next document's 1, and
    // `rangeTree.tailSet` (cut after 5 missed [0, 1]) no longer finds [0, 1] for that 1.
    int tx = id % 200;
    if (tx == 50 || tx >= 56) s.add("SortedNumericDocValuesField tx 100 1");
    if (tx == 54) {
      s.add("SortedNumericDocValuesField tx 0 1");
      s.add("SortedNumericDocValuesField tx 5 1");
    }
    if (tx == 55) s.add("SortedNumericDocValuesField tx 1 1");
    if (r.nextInt(5) != 0) s.add("NumericDocValuesField n " + (r.nextInt(101) - 50) + " 0");
    int nsx = r.nextInt(3);
    for (int i = 0; i < nsx; i++) {
      s.add("SortedNumericDocValuesField snx " + (id % 200 + r.nextInt(20)) + " 1");
    }
    int nss = r.nextInt(4);
    for (int i = 0; i < nss; i++) s.add("SortedSetDocValuesField ss " + term(r) + " 0");
    if (r.nextInt(5) != 0) s.add("SortedDocValuesField s " + term(r) + " 0");
    if (r.nextInt(3) != 0) s.add("SortedSetDocValuesField ssx t" + (10 + (id / 200) * 10 + r.nextInt(10)) + " 1");
    return s;
  }

  // --- queries -----------------------------------------------------------------------------

  /**
   * Builds one query from its spec. Ranges are {@code lo:hi} separated by {@code ;}; a
   * multi-dimensional bound is comma separated.
   */
  static Query query(String spec) {
    String[] a = spec.split(" ");
    String f = a.length > 1 ? a[1] : "";
    switch (a[0]) {
      case "boost":
        return new BoostQuery(
            query(String.join(" ", Arrays.copyOfRange(a, 2, a.length))), Float.parseFloat(a[1]));
      case "HalfRange":
        return HalfFloatPoint.newRangeQuery(f, floats(a[2]), floats(a[3]));
      case "HalfExact":
        return HalfFloatPoint.newExactQuery(f, Float.parseFloat(a[2]));
      case "HalfSet":
        return HalfFloatPoint.newSetQuery(f, a.length > 2 ? floats(a[2]) : new float[0]);
      case "HalfSetList":
        {
          List<Float> l = new ArrayList<>();
          for (float v : floats(a[2])) l.add(v);
          return HalfFloatPoint.newSetQuery(f, l);
        }
      case "BigRange":
        return BigIntegerPoint.newRangeQuery(f, bigs(a[2]), bigs(a[3]));
      case "BigRange1":
        return BigIntegerPoint.newRangeQuery(f, new BigInteger(a[2]), new BigInteger(a[3]));
      case "BigExact":
        return BigIntegerPoint.newExactQuery(f, new BigInteger(a[2]));
      case "BigSet":
        return BigIntegerPoint.newSetQuery(f, a.length > 2 ? bigs(a[2]) : new BigInteger[0]);
      case "BBox":
        {
          double[] d = doubles(a[3]);
          switch (a[2]) {
            case "INTERSECTS":
              return LatLonBoundingBox.newIntersectsQuery(f, d[0], d[1], d[2], d[3]);
            case "WITHIN":
              return LatLonBoundingBox.newWithinQuery(f, d[0], d[1], d[2], d[3]);
            case "CONTAINS":
              return LatLonBoundingBox.newContainsQuery(f, d[0], d[1], d[2], d[3]);
            case "CROSSES":
              return LatLonBoundingBox.newCrossesQuery(f, d[0], d[1], d[2], d[3]);
            default:
              throw new IllegalArgumentException(spec);
          }
        }
      case "IntMR":
        {
          IntPointMultiRangeBuilder b = new IntPointMultiRangeBuilder(f, Integer.parseInt(a[2]));
          for (String r : ranges(a)) b.add(ints(r.split(":")[0]), ints(r.split(":")[1]));
          return b.build();
        }
      case "LongMR":
        {
          LongPointMultiRangeBuilder b = new LongPointMultiRangeBuilder(f, Integer.parseInt(a[2]));
          for (String r : ranges(a)) b.add(longs(r.split(":")[0]), longs(r.split(":")[1]));
          return b.build();
        }
      case "FloatMR":
        {
          FloatPointMultiRangeBuilder b = new FloatPointMultiRangeBuilder(f, Integer.parseInt(a[2]));
          for (String r : ranges(a)) b.add(floats(r.split(":")[0]), floats(r.split(":")[1]));
          return b.build();
        }
      case "DoubleMR":
        {
          DoublePointMultiRangeBuilder b =
              new DoublePointMultiRangeBuilder(f, Integer.parseInt(a[2]));
          for (String r : ranges(a)) b.add(doubles(r.split(":")[0]), doubles(r.split(":")[1]));
          return b.build();
        }
      case "RawMR":
        {
          // MultiRangeQuery.Builder.add(byte[], byte[]) with IntPoint-packed bounds, unchecked.
          MultiRangeQuery.Builder b =
              new IntPointMultiRangeBuilder(f, Integer.parseInt(a[2]));
          for (String r : ranges(a)) {
            b.add(IntPoint.pack(ints(r.split(":")[0])).bytes, IntPoint.pack(ints(r.split(":")[1])).bytes);
          }
          return b.build();
        }
      case "SNMR":
        {
          DocValuesMultiRangeQuery.SortedNumericStabbingBuilder b =
              new DocValuesMultiRangeQuery.SortedNumericStabbingBuilder(f);
          for (String r : ranges(a, 2)) {
            String[] p = r.split(":");
            b.add(Long.parseLong(p[0]), Long.parseLong(p[1]));
          }
          return b.build();
        }
      case "SSMR":
        {
          DocValuesMultiRangeQuery.SortedSetStabbingBuilder b =
              new DocValuesMultiRangeQuery.SortedSetStabbingBuilder(f);
          for (String r : ranges(a, 2)) {
            String[] p = r.split(":", -1);
            if (p.length == 1) b.add(new BytesRef(p[0]));
            else b.add(new BytesRef(p[0]), new BytesRef(p[1]));
          }
          return b.build();
        }
      default:
        throw new IllegalArgumentException(spec);
    }
  }

  static String[] ranges(String[] a) {
    return ranges(a, 3);
  }

  static String[] ranges(String[] a, int at) {
    return a.length > at ? a[at].split(";") : new String[0];
  }

  static List<String> querySpecs(Random r) {
    List<String> q = new ArrayList<>();
    q.add("HalfRange h -10.5 10.5");
    q.add("HalfRange h 0 0");
    q.add("HalfRange h -0 -0");
    q.add("HalfRange h 1 1.001");
    q.add("HalfRange h 65504 Infinity");
    q.add("HalfRange h -Infinity -100");
    q.add("HalfRange h 1e-8 1e-7");
    q.add("HalfRange h 5 -5");
    q.add("HalfRange h2 -50,-50 50,50");
    q.add("HalfRange h2 0,-200 200,0");
    q.add("HalfExact h 1.5");
    q.add("HalfExact h 65520");
    q.add("HalfExact h 1.0009765625");
    q.add("HalfSet h 1.5,-1,0,65504,3.14159,1e-8");
    q.add("HalfSet h");
    q.add("HalfSetList h 2.5,-2.71828,1000.4");
    q.add("HalfRange missing 0 1");
    q.add("BigRange1 b -50 50");
    q.add("BigRange1 b 0 0");
    q.add(
        "BigRange1 b "
            + BigIntegerPoint.MIN_VALUE
            + " "
            + BigIntegerPoint.MAX_VALUE);
    q.add("BigRange1 b " + BigIntegerPoint.MIN_VALUE + " -1000");
    q.add("BigRange1 b 1000 " + BigIntegerPoint.MAX_VALUE);
    q.add("BigRange b2 -100,-100 100,100");
    q.add("BigRange b2 0," + BigIntegerPoint.MIN_VALUE + " " + BigIntegerPoint.MAX_VALUE + ",0");
    q.add("BigExact b 7");
    q.add("BigExact b " + BigIntegerPoint.MAX_VALUE);
    q.add("BigSet b 7,-3,99,0," + BigIntegerPoint.MIN_VALUE);
    q.add("BigSet b");
    for (String type : new String[] {"INTERSECTS", "WITHIN", "CONTAINS", "CROSSES"}) {
      q.add("BBox bb " + type + " -10,-20,30,40");
      q.add("BBox bb " + type + " 0,0,5,5");
      q.add("BBox bb " + type + " -90,-180,90,180");
      q.add("BBox bb " + type + " 45.5,100.25,46.5,101.25");
    }
    q.add("IntMR i 1 -100:100");
    q.add("IntMR i 1 -100:100;50:300;-700:-650;900:1000");
    q.add("IntMR i 1 -1000:-990;-1000:-995;0:0;0:5;3:4");
    q.add("IntMR i 1 5:5");
    q.add("IntMR i 1 -2147483648:2147483647;0:1");
    q.add("IntMR i2 2 -10,-10:10,10;50,-100:100,-50;-100,50:-50,100");
    q.add("IntMR i2 2 -100,-100:100,100");
    q.add("IntMR i2 2 0,0:0,0;1,1:1,1;2,2:3,3");
    q.add("RawMR i 1 100:-100;-5:5");
    q.add("LongMR l 1 -42949672960:42949672960;100000000000:200000000000");
    q.add("LongMR l 1 -9223372036854775808:0");
    q.add("FloatMR f 1 -10.5:10.5;100:120;-125:-100;-0.0:-0.0");
    q.add("FloatMR f 1 -Infinity:Infinity");
    q.add("DoubleMR d 1 -5.5:5.5;20:30;-62.5:-50");
    q.add("DoubleMR d2 2 -5,-5:5,5;10,10:20,20;-25,-25:-20,-20;0,0:25,25");
    q.add("boost 2.5 IntMR i 1 -100:100;50:300");
    // Random multi-range queries.
    for (int k = 0; k < 20; k++) {
      int n = 2 + r.nextInt(8);
      StringBuilder sb = new StringBuilder("IntMR i 1 ");
      for (int j = 0; j < n; j++) {
        int lo = r.nextInt(2001) - 1000;
        int hi = lo + r.nextInt(150);
        if (j > 0) sb.append(';');
        sb.append(lo).append(':').append(hi);
      }
      q.add(sb.toString());
    }
    for (int k = 0; k < 10; k++) {
      int n = 2 + r.nextInt(8);
      StringBuilder sb = new StringBuilder("IntMR i2 2 ");
      for (int j = 0; j < n; j++) {
        int lo1 = r.nextInt(201) - 100, lo2 = r.nextInt(201) - 100;
        if (j > 0) sb.append(';');
        sb.append(lo1).append(',').append(lo2).append(':')
            .append(lo1 + r.nextInt(40)).append(',').append(lo2 + r.nextInt(40));
      }
      q.add(sb.toString());
    }
    // Doc-values multi-range queries.
    q.add("SNMR sn");
    q.add("SNMR sn -10:10");
    q.add("SNMR sn -10:10;20:30");
    q.add("SNMR sn -10:10;5:20;40:45;0:0;45:45;-50:-50");
    q.add("SNMR sn 3:3;3:3;7:7;10:2");
    q.add("SNMR sn -9223372036854775808:-40;40:9223372036854775807");
    q.add("SNMR sn -9223372036854775808:1;5:6");
    q.add("SNMR sn 9223372036854775807:-1;-5:5");
    q.add("SNMR sn 1:5;5:9;9:9");
    q.add("SNMR sn -100:-60;60:100");
    q.add("SNMR n -10:10;20:30;-50:-45");
    q.add("SNMR n 0:0;1:1;2:2");
    q.add("SNMR snx 0:5;100:110");
    q.add("SNMR snx 500:600;700:800");
    q.add("SNMR snx 30:35;150:160;210:215");
    q.add("SNMR nx 0:5;100:110");
    q.add("SNMR nx 30:35;150:160;210:215");
    q.add("SNMR tx 0:1;10:20");
    q.add("SNMR tx 0:1;10:20;100:100");
    q.add("SNMR missing 1:2;3:4");
    q.add("boost 3 SNMR sn -10:10;20:30");
    q.add("SSMR ss");
    q.add("SSMR ss t05:t10");
    q.add("SSMR ss t05:t10;t20:t25");
    q.add("SSMR ss t05:t10;t08:t15;t30;t31;t32:t32");
    q.add("SSMR ss t10:t05;t50:t60");
    q.add("SSMR ss a:b;t00:t01;t39:z");
    q.add("SSMR ss a:t00;t395:u;t205:t215");
    q.add("SSMR ss t05;t06;t07");
    q.add("SSMR ss t051:t052;t11:t12");
    q.add("SSMR s t05:t10;t20:t25");
    q.add("SSMR s t00:t01;t38:t39");
    q.add("SSMR ssx t10:t12;t25:t27");
    q.add("SSMR ssx t40:t45;t46:t50");
    q.add("SSMR missing a:b;c:d");
    q.add("boost 0.5 SSMR ss t05:t10;t20:t25");
    for (int k = 0; k < 10; k++) {
      int n = 2 + r.nextInt(6);
      StringBuilder sb = new StringBuilder("SNMR sn ");
      for (int j = 0; j < n; j++) {
        long lo = r.nextInt(121) - 60;
        long hi = lo + r.nextInt(15) - 2;
        if (j > 0) sb.append(';');
        sb.append(lo).append(':').append(hi);
      }
      q.add(sb.toString());
    }
    for (int k = 0; k < 10; k++) {
      int n = 2 + r.nextInt(6);
      StringBuilder sb = new StringBuilder("SSMR ss ");
      for (int j = 0; j < n; j++) {
        int lo = r.nextInt(44) - 2;
        int hi = lo + r.nextInt(8) - 1;
        if (j > 0) sb.append(';');
        sb.append(String.format("t%02d", Math.max(lo, 0)));
        if (r.nextInt(4) != 0) sb.append(':').append(String.format("t%02d", Math.max(hi, 0)));
      }
      q.add(sb.toString());
    }
    return q;
  }

  static final String[] ERRORS = {
    "BBox bb INTERSECTS 10,40,20,30",
    "BBox bb WITHIN 30,10,20,40",
    "BBox bb CONTAINS 95,10,96,20",
    "BBox bb CROSSES -10,-190,10,0",
    "IntMR i 1 1,2:3,4",
    "IntMR i 1 5:4",
    "LongMR l 2 1:2",
    "FloatMR f 1 2:1",
    "DoubleMR d 1 1:NaN",
    "IntMR i2 1 0:1;2:3",
    "IntMR i 2 0,0:1,1;2,2:3,3",
    "LongMR i 1 0:1;2:3",
    "IntMR i 1",
    "SNMR sn 10:2;30:20",
    "HalfRange h 0,0 1",
    "HalfRange h 0,0 1,1",
    "HalfRange h2 0 1",
  };

  static final String[] NEAREST = {
    "1 0,0,0",
    "5 0,0,0",
    "10 1.5,-2.25,3",
    "25 -10,10,-10",
    "3 100,100,100",
    "1000 0,0,0",
    "7 5,5",
    "4 -3.5",
  };

  // --- output helpers ----------------------------------------------------------------------

  /** Every match with its score, as GenDocumentFields.hits. */
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

  static final int LARGE_DOCS = 9000;

  static final String[] LARGE_QUERIES = {
    // Dense, skip-indexed, three skip blocks: the middle one's values miss every range.
    "SNMR kx 100:200;8500:8600",
    "SNMR kx 0:1500;8500:8600",
    "SNMR kx 4000:4100;8191:8192",
    "SNMR kx 5000:5100",
    "SNMR dx 0:5;100:110",
    "SNMR dx 0:5;100:110;990:999",
    "SNMR dx 2000:3000",
    "SNMR gx 0:5;100:110",
    "SNMR gx 1:1;500:900",
    "SNMR hx 0:3;200:203",
    "SNMR hx 10:12;250:299",
    "SNMR ux 0:5;100:110",
    "SNMR vx 0:5;100:110",
    "SNMR vx 0:3000",
    // Over 1,000 hits: where the collector saturates decides the total hits. Two ranges each,
    // since one is the plain doc-values range query.
    "SNMR dx 0:500;502:999",
    "SNMR dx 0:600;700:999",
    "SNMR gx 0:3;5:6000",
    "SNMR hx 0:3;5:6000",
    "SNMR ux 0:500;502:999",
    // One range: the builder returns core's SortedNumericDocValuesField.newSlowRangeQuery,
    // scored by ConstantScoreScorerSupplier -- a match-all over a skip index that covers the
    // whole range, YES skip blocks whose docIDRunEnd shapes the windows, MAYBE ones, no skip
    // index (dense and sparse), a multi-valued field.
    "SNMR dx 0:999",
    "SNMR dx 100:600",
    "SNMR kx 0:4500",
    "SNMR kx 1000:8999",
    "SNMR kx 2000:2100",
    "SNMR ux 0:999",
    "SNMR vx 0:999",
    "SNMR gx 0:6000",
  };

  /**
   * One segment of 9,000 documents, every ninth deleted: large enough for
   * DenseConjunctionBulkScorer's windows (at least 2,048 documents), with sparse skip-indexed
   * fields whose stale reads depend on the order the bulk scorer confirms and masks candidates.
   * Per query: search(q, 10)'s total hits and relation, every hit, and count(q).
   */
  static void large(Path root) throws IOException {
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    StringBuilder docsOut = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < LARGE_DOCS; i++) {
          List<String> specs = new ArrayList<>();
          specs.add("id " + i);
          specs.add("SortedNumericDocValuesField dx " + (i % 1000) + " 1");
          if (i % 4 == 0) {
            specs.add("SortedNumericDocValuesField gx " + (i % 1000) + " 1");
            specs.add("SortedNumericDocValuesField gx " + (i % 1000 + 1) + " 1");
          }
          if (i % 7 == 0) specs.add("SortedNumericDocValuesField hx " + (i % 300) + " 1");
          // A dense tail: a stale read runs at most a gap's worth of values ahead, and past a
          // field's last value Lucene reads past the column (an EOFException, or padding decoded
          // as a value).
          if (i >= LARGE_DOCS - 20) {
            specs.add("SortedNumericDocValuesField gx 5000 1");
            specs.add("SortedNumericDocValuesField hx 5000 1");
          }
          specs.add("SortedNumericDocValuesField kx " + i + " 1");
          specs.add("SortedNumericDocValuesField ux " + (i % 1000) + " 0");
          if (i % 40 == 0) specs.add("SortedNumericDocValuesField vx " + (i % 1000) + " 0");
          Document doc = new Document();
          docsOut.append(i);
          for (String s : specs) {
            docsOut.append('\t').append(s);
            doc.add(field(s));
          }
          docsOut.append('\n');
          w.addDocument(doc);
        }
        w.commit();
        for (int i = 0; i < LARGE_DOCS; i += 9) w.deleteDocuments(new Term("id", Integer.toString(i)));
        w.commit();
      }
      Files.writeString(root.resolve("docs.tsv"), docsOut);
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != 1) throw new AssertionError("segments");
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        StringBuilder out = new StringBuilder();
        for (String spec : LARGE_QUERIES) {
          Query q = query(spec);
          TopDocs top = searcher.search(q, 10);
          out.append(spec).append('\t').append(top.totalHits.value()).append('\t')
              .append(top.totalHits.relation()).append('\t');
          for (ScoreDoc sd : top.scoreDocs) out.append(sd.doc).append(',');
          out.append('\t').append(searcher.count(q)).append('\t').append(hits(searcher, q))
              .append('\n');
        }
        Files.writeString(root.resolve("queries.tsv"), out);
      }
    }
  }

  static String fbits(float f) {
    return Integer.toHexString(Float.floatToRawIntBits(f));
  }

  static void clean(Path root) throws IOException {
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        for (Path p : walk.sorted((x, y) -> y.compareTo(x)).toList()) Files.delete(p);
      }
    }
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("sandbox_document");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);

    Random r = new Random(20261019L);
    StringBuilder docsOut = new StringBuilder();
    StringBuilder packedOut = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
            int id = seg * DOCS_PER_SEGMENT + i;
            List<String> specs = docSpecs(r, id);
            docsOut.append(id);
            Document doc = new Document();
            for (String s : specs) {
              docsOut.append('\t').append(s);
              IndexableField f = field(s);
              if (s.startsWith("HalfFloatPoint")
                  || s.startsWith("BigIntegerPoint")
                  || s.startsWith("LatLonBoundingBox")) {
                BytesRef b = f.binaryValue();
                packedOut.append(s).append('\t')
                    .append(HEX.formatHex(b.bytes, b.offset, b.offset + b.length)).append('\n');
              }
              doc.add(f);
            }
            docsOut.append('\n');
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int i = 0; i < DOCS_PER_SEGMENT; i += 13) {
          w.deleteDocuments(new Term("id", Integer.toString(i)));
          w.deleteDocuments(new Term("id", Integer.toString(2 * DOCS_PER_SEGMENT + i)));
        }
        w.commit();
      }
      Files.writeString(root.resolve("docs.tsv"), docsOut);
      Files.writeString(root.resolve("packed.tsv"), packedOut);

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        if (reader.leaves().size() != SEGMENTS) throw new AssertionError("segments");
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        StringBuilder q = new StringBuilder();
        for (String spec : querySpecs(new Random(77L))) {
          q.append(spec).append('\t').append(hits(searcher, query(spec))).append('\n');
        }
        Files.writeString(root.resolve("queries.tsv"), q);

        StringBuilder e = new StringBuilder();
        for (String spec : ERRORS) {
          String msg;
          try {
            searcher.search(query(spec), 10);
            msg = "OK";
          } catch (IllegalArgumentException ex) {
            msg = ex.getMessage();
          } catch (RuntimeException ex) {
            msg = ex.getClass().getSimpleName();
          }
          e.append(spec).append('\t').append(msg).append('\n');
        }
        for (String n : new String[] {"0", "-3"}) {
          String spec = "nearest " + n + " 0,0,0";
          String msg;
          try {
            FloatPointNearestNeighbor.nearest(searcher, "v", Integer.parseInt(n), 0f, 0f, 0f);
            msg = "OK";
          } catch (IllegalArgumentException ex) {
            msg = ex.getMessage();
          }
          e.append(spec).append('\t').append(msg).append('\n');
        }
        Files.writeString(root.resolve("errors.tsv"), e);

        StringBuilder nn = new StringBuilder();
        for (String spec : NEAREST) {
          String[] p = spec.split(" ");
          TopFieldDocs td =
              FloatPointNearestNeighbor.nearest(searcher, "v", Integer.parseInt(p[0]), floats(p[1]));
          nn.append(spec).append('\t').append(td.totalHits.value());
          for (ScoreDoc sd : td.scoreDocs) {
            nn.append('\t').append(sd.doc).append(':')
                .append(fbits((Float) ((FieldDoc) sd).fields[0]));
          }
          nn.append('\n');
        }
        Files.writeString(root.resolve("nearest.tsv"), nn);
      }
    }

    large(root.resolve("large"));

    // HalfFloatPoint's encoding, value by value.
    StringBuilder half = new StringBuilder();
    List<Float> vals = new ArrayList<>();
    for (float f : HALF_SPECIALS) vals.add(f);
    vals.add(Float.NaN);
    vals.add(Float.intBitsToFloat(0x7fc00001));
    vals.add(Float.intBitsToFloat(0xffc00000));
    Random hr = new Random(5L);
    for (int i = 0; i < 2000; i++) vals.add(Float.intBitsToFloat(hr.nextInt()));
    for (int i = 0; i < 500; i++) vals.add((hr.nextInt(2_000_001) - 1_000_000) / 1000f);
    for (int s = Short.MIN_VALUE; s <= Short.MAX_VALUE; s += 7) {
      vals.add(HalfFloatPoint.sortableShortToHalfFloat((short) s));
    }
    byte[] enc = new byte[2];
    for (float f : vals) {
      short s = HalfFloatPoint.halfFloatToSortableShort(f);
      HalfFloatPoint.encodeDimension(f, enc, 0);
      half.append(fbits(f)).append('\t').append(s).append('\t')
          .append(fbits(HalfFloatPoint.sortableShortToHalfFloat(s))).append('\t')
          .append(fbits(HalfFloatPoint.nextUp(f))).append('\t')
          .append(fbits(HalfFloatPoint.nextDown(f))).append('\t')
          .append(HEX.formatHex(enc)).append('\t')
          .append(fbits(HalfFloatPoint.decodeDimension(enc, 0))).append('\n');
    }
    Files.writeString(root.resolve("half.tsv"), half);

    StringBuilder big = new StringBuilder();
    Random br = new Random(9L);
    List<BigInteger> bv = new ArrayList<>();
    bv.add(BigIntegerPoint.MIN_VALUE);
    bv.add(BigIntegerPoint.MAX_VALUE);
    bv.add(BigInteger.ZERO);
    bv.add(BigInteger.ONE);
    bv.add(BigInteger.ONE.negate());
    for (int i = 0; i < 300; i++) bv.add(new BigInteger(128, br).subtract(BigInteger.ONE.shiftLeft(127)));
    for (int i = 0; i < 100; i++) bv.add(BigInteger.valueOf(br.nextLong()));
    byte[] be = new byte[BigIntegerPoint.BYTES];
    for (BigInteger v : bv) {
      BigIntegerPoint.encodeDimension(v, be, 0);
      if (!BigIntegerPoint.decodeDimension(be, 0).equals(v)) throw new AssertionError(v);
      big.append(v).append('\t').append(HEX.formatHex(be)).append('\n');
    }
    Files.writeString(root.resolve("big.tsv"), big);
  }
}
