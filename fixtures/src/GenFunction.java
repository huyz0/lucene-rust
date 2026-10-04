import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.DoublePredicate;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleDocValuesField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatDocValuesField;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.queries.function.FunctionMatchQuery;
import org.apache.lucene.queries.function.FunctionQuery;
import org.apache.lucene.queries.function.FunctionRangeQuery;
import org.apache.lucene.queries.function.FunctionScoreQuery;
import org.apache.lucene.queries.function.FunctionValues;
import org.apache.lucene.queries.function.IndexReaderFunctions;
import org.apache.lucene.queries.function.ValueSource;
import org.apache.lucene.queries.function.valuesource.*;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.SortedNumericSelector;
import org.apache.lucene.search.SortedSetSelector;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.grouping.GroupDocs;
import org.apache.lucene.search.grouping.GroupingSearch;
import org.apache.lucene.search.grouping.TopGroups;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefBuilder;
import org.apache.lucene.util.NumericUtils;
import org.apache.lucene.util.mutable.MutableValue;

/**
 * M10 T10.5's function-query fixture: {@code function/index}, a four-segment index with deletions
 * in two segments, missing and multi-valued doc values, and every kind of field a value source
 * reads.
 *
 * <p>Fields: {@code id} (StringField, docs only), {@code body} (text, with norms), {@code i}/{@code
 * l} (NUMERIC int/long), {@code f}/{@code d} (float/double doc values, sometimes {@code NaN}, an
 * infinity or {@code -0.0}), {@code mi}/{@code ml} (SORTED_NUMERIC, zero to three values), {@code
 * mf}/{@code md} (SORTED_NUMERIC sortable float/double), {@code s} (SORTED), {@code ss}
 * (SORTED_SET), {@code b} (BINARY, sometimes empty), {@code e} (NUMERIC enum ordinals), {@code k}
 * (SORTED words, for {@code joindf}), {@code fv}/{@code bv} (float/byte vectors).
 *
 * <p>{@code values.tsv}: for each value source (the spec grammar is the Rust test's, {@code
 * function_fixtures.rs}), every document of every segment (deleted ones included): {@code exists},
 * {@code byteVal}, {@code shortVal}, {@code floatVal} and {@code doubleVal} bits, {@code intVal},
 * {@code longVal}, {@code boolVal}, {@code strVal}, {@code objectVal}, {@code ordVal}, the vector
 * getters, {@code toString(doc)} and the {@code ValueFiller}'s value -- each, or the exception it
 * threw. Searched with {@code ClassicSimilarity} (so {@code tf}/{@code idf}/{@code norm} have a
 * {@code TFIDFSimilarity}), except the specs marked {@code bm25}.
 *
 * <p>{@code searches.tsv}: {@code FunctionQuery}, {@code FunctionRangeQuery}, {@code
 * FunctionScoreQuery} (with {@code boostByValue}, {@code boostByQuery}, the {@code
 * IndexReaderFunctions}), {@code FunctionMatchQuery}, the function queries inside booleans, and
 * sorts by {@code ValueSource.getSortField}: every hit with its score bits, then the explanation of
 * three documents.
 *
 * <p>Usage: {@code java GenFunction <fixtures-data-dir>}.
 */
public class GenFunction {
  static final String[] WORDS = {"red", "blue", "green", "fast", "slow", "big", "small", "old"};

  static void clean(Path root) throws IOException {
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        for (Path p : walk.sorted(Comparator.reverseOrder()).toList()) Files.delete(p);
      }
    }
  }

  static String word(Random r) {
    int i = (int) (Math.pow(r.nextDouble(), 1.5) * WORDS.length);
    return WORDS[Math.min(i, WORDS.length - 1)];
  }

  static float odd(Random r, float v) {
    return switch (r.nextInt(30)) {
      case 0 -> Float.NaN;
      case 1 -> Float.POSITIVE_INFINITY;
      case 2 -> -0.0f;
      default -> v;
    };
  }

  static Document doc(Random r, int id) {
    Document d = new Document();
    d.add(new StringField("id", "d" + id, Field.Store.NO));
    int n = 1 + r.nextInt(7);
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) {
      if (i > 0) b.append(' ');
      b.append(word(r));
    }
    d.add(new TextField("body", b.toString(), Field.Store.NO));
    if (r.nextInt(6) != 0) d.add(new NumericDocValuesField("i", r.nextInt(46) - 6));
    if (r.nextInt(6) != 0) {
      long v = r.nextBoolean() ? r.nextInt(1000) - 300 : (r.nextLong() >> r.nextInt(40));
      d.add(new NumericDocValuesField("l", v));
    }
    if (r.nextInt(6) != 0) {
      d.add(new FloatDocValuesField("f", odd(r, (r.nextInt(200) - 60) / 8f + r.nextFloat())));
    }
    if (r.nextInt(6) != 0) {
      double v = (r.nextInt(200) - 60) / 3.0 + r.nextDouble();
      float o = odd(r, 1f);
      d.add(new DoubleDocValuesField("d", o != 1f ? o : v));
    }
    for (int i = r.nextInt(4); i > 0; i--) d.add(new SortedNumericDocValuesField("mi", r.nextInt(50) - 10));
    for (int i = r.nextInt(4); i > 0; i--) {
      d.add(new SortedNumericDocValuesField("ml", r.nextLong() >> r.nextInt(50)));
    }
    for (int i = r.nextInt(4); i > 0; i--) {
      float v = (r.nextInt(80) - 20) / 4f;
      d.add(new SortedNumericDocValuesField("mf", NumericUtils.floatToSortableInt(v)));
    }
    for (int i = r.nextInt(4); i > 0; i--) {
      double v = (r.nextInt(80) - 20) / 7.0;
      d.add(new SortedNumericDocValuesField("md", NumericUtils.doubleToSortableLong(v)));
    }
    if (r.nextInt(5) != 0) d.add(new SortedDocValuesField("s", new BytesRef(word(r) + r.nextInt(3))));
    for (int i = r.nextInt(4); i > 0; i--) {
      d.add(new SortedSetDocValuesField("ss", new BytesRef(word(r))));
    }
    if (r.nextInt(5) != 0) {
      d.add(new BinaryDocValuesField("b", new BytesRef(r.nextInt(7) == 0 ? "" : "bin" + r.nextInt(9))));
    }
    if (r.nextInt(5) != 0) d.add(new NumericDocValuesField("e", r.nextInt(6)));
    if (r.nextInt(4) != 0) d.add(new SortedDocValuesField("k", new BytesRef(WORDS[r.nextInt(WORDS.length)])));
    if (r.nextInt(4) != 0) {
      float[] v = new float[4];
      for (int i = 0; i < 4; i++) v[i] = (r.nextInt(40) - 20) / 10f;
      if (v[0] == 0 && v[1] == 0 && v[2] == 0 && v[3] == 0) v[0] = 1;
      d.add(new KnnFloatVectorField("fv", v, VectorSimilarityFunction.EUCLIDEAN));
    }
    if (r.nextInt(4) != 0) {
      byte[] v = new byte[4];
      for (int i = 0; i < 4; i++) v[i] = (byte) (r.nextInt(200) - 100);
      d.add(new KnnByteVectorField("bv", v, VectorSimilarityFunction.EUCLIDEAN));
    }
    return d;
  }

  static IndexWriterConfig config() {
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setUseCompoundFile(false);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setRAMBufferSizeMB(256);
    return cfg;
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("function");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(0x10_5_2026_1004L);
    try (Directory dir = FSDirectory.open(indexDir);
        IndexWriter w = new IndexWriter(dir, config())) {
      int id = 0;
      int[] sizes = {26, 20, 24, 10};
      for (int seg = 0; seg < sizes.length; seg++) {
        for (int i = 0; i < sizes[seg]; i++, id++) {
          w.addDocument(doc(r, id));
        }
        w.commit();
        if (seg == 1 || seg == 2) {
          for (int k = 0; k < 5; k++) {
            w.deleteDocuments(new Term("id", "d" + (id - 1 - r.nextInt(sizes[seg]))));
          }
          w.commit();
        }
      }
    }
    StringBuilder values = new StringBuilder();
    StringBuilder searches = new StringBuilder();
    StringBuilder groups = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 4) throw new AssertionError("segments");
      GenFunction g = new GenFunction(reader);
      g.values(values);
      g.searches(searches);
      g.groups(groups);
    }
    Files.writeString(root.resolve("values.tsv"), values.toString(), StandardCharsets.UTF_8);
    Files.writeString(root.resolve("searches.tsv"), searches.toString(), StandardCharsets.UTF_8);
    Files.writeString(root.resolve("groups.tsv"), groups.toString(), StandardCharsets.UTF_8);
  }

  final DirectoryReader reader;
  final IndexSearcher bm25;
  final IndexSearcher classic;

  GenFunction(DirectoryReader reader) {
    this.reader = reader;
    this.bm25 = new IndexSearcher(reader);
    bm25.setQueryCache(null);
    this.classic = new IndexSearcher(reader);
    classic.setQueryCache(null);
    classic.setSimilarity(new ClassicSimilarity());
  }

  // ---------------------------------------------------------------------------------------------
  // The spec grammar: name(arg,arg,...), args split at top-level commas.
  // ---------------------------------------------------------------------------------------------

  static List<String> split(String s) {
    List<String> out = new ArrayList<>();
    int depth = 0, start = 0;
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c == '(') depth++;
      else if (c == ')') depth--;
      else if (c == ',' && depth == 0) {
        out.add(s.substring(start, i));
        start = i + 1;
      }
    }
    if (start < s.length() || !out.isEmpty()) out.add(s.substring(start));
    return out;
  }

  static String name(String spec) {
    int p = spec.indexOf('(');
    return p < 0 ? spec : spec.substring(0, p);
  }

  static List<String> args(String spec) {
    int p = spec.indexOf('(');
    if (p < 0) return List.of();
    return split(spec.substring(p + 1, spec.length() - 1));
  }

  /** A query: {@code t:w}, {@code or:a:b}, {@code and:a:b}, {@code all}. */
  static Query query(String spec) {
    String[] p = spec.split(":");
    return switch (p[0]) {
      case "all" -> new MatchAllDocsQuery();
      case "t" -> new TermQuery(new Term("body", p[1]));
      case "or", "and" -> {
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        BooleanClause.Occur o = p[0].equals("or") ? BooleanClause.Occur.SHOULD : BooleanClause.Occur.MUST;
        b.add(new TermQuery(new Term("body", p[1])), o);
        b.add(new TermQuery(new Term("body", p[2])), o);
        yield b.build();
      }
      default -> throw new IllegalArgumentException(spec);
    };
  }

  static ValueSource[] vsList(List<String> a) {
    ValueSource[] out = new ValueSource[a.size()];
    for (int i = 0; i < out.length; i++) out[i] = vs(a.get(i));
    return out;
  }

  static ValueSource vs(String spec) {
    List<String> a = args(spec);
    return switch (name(spec)) {
      case "int" -> new IntFieldSource(a.get(0));
      case "long" -> new LongFieldSource(a.get(0));
      case "float" -> new FloatFieldSource(a.get(0));
      case "double" -> new DoubleFieldSource(a.get(0));
      case "mint" -> new MultiValuedIntFieldSource(a.get(0), SortedNumericSelector.Type.valueOf(a.get(1)));
      case "mlong" -> new MultiValuedLongFieldSource(a.get(0), SortedNumericSelector.Type.valueOf(a.get(1)));
      case "mfloat" -> new MultiValuedFloatFieldSource(a.get(0), SortedNumericSelector.Type.valueOf(a.get(1)));
      case "mdouble" -> new MultiValuedDoubleFieldSource(a.get(0), SortedNumericSelector.Type.valueOf(a.get(1)));
      case "bytes" -> new BytesRefFieldSource(a.get(0));
      case "sset" -> new SortedSetFieldSource(a.get(0), SortedSetSelector.Type.valueOf(a.get(1)));
      case "enum" -> {
        Map<Integer, String> i2s = new HashMap<>();
        Map<String, Integer> s2i = new HashMap<>();
        String[] names = {"zero", "one", "two", "three"};
        for (int i = 0; i < names.length; i++) {
          i2s.put(i, names[i]);
          s2i.put(names[i], i);
        }
        yield new EnumFieldSource(a.get(0), i2s, s2i);
      }
      case "joindf" -> new JoinDocFreqValueSource(a.get(0), a.get(1));
      case "const" -> new ConstValueSource(Float.parseFloat(a.get(0)));
      case "dconst" -> new DoubleConstValueSource(Double.parseDouble(a.get(0)));
      case "literal" -> new LiteralValueSource(a.get(0));
      case "sum" -> new SumFloatFunction(vsList(a));
      case "product" -> new ProductFloatFunction(vsList(a));
      case "max" -> new MaxFloatFunction(vsList(a));
      case "min" -> new MinFloatFunction(vsList(a));
      case "div" -> new DivFloatFunction(vs(a.get(0)), vs(a.get(1)));
      case "pow" -> new PowFloatFunction(vs(a.get(0)), vs(a.get(1)));
      case "linear" -> new LinearFloatFunction(vs(a.get(0)), Float.parseFloat(a.get(1)), Float.parseFloat(a.get(2)));
      case "recip" ->
          new ReciprocalFloatFunction(
              vs(a.get(0)), Float.parseFloat(a.get(1)), Float.parseFloat(a.get(2)), Float.parseFloat(a.get(3)));
      case "map" ->
          new RangeMapFloatFunction(
              vs(a.get(0)),
              Float.parseFloat(a.get(1)),
              Float.parseFloat(a.get(2)),
              vs(a.get(3)),
              a.get(4).equals("null") ? null : vs(a.get(4)));
      case "scale" -> new ScaleFloatFunction(vs(a.get(0)), Float.parseFloat(a.get(1)), Float.parseFloat(a.get(2)));
      case "def" -> new DefFunction(Arrays.asList(vsList(a)));
      case "if" -> new IfFunction(vs(a.get(0)), vs(a.get(1)), vs(a.get(2)));
      case "exists" ->
          new SimpleBoolFunction(vs(a.get(0))) {
            @Override
            protected String name() {
              return "exists";
            }

            @Override
            protected boolean func(int doc, FunctionValues vals) throws IOException {
              return vals.exists(doc);
            }
          };
      case "and", "or" -> {
        boolean and = name(spec).equals("and");
        yield new MultiBoolFunction(Arrays.asList(vsList(a))) {
          @Override
          protected String name() {
            return and ? "and" : "or";
          }

          @Override
          protected boolean func(int doc, FunctionValues[] vals) throws IOException {
            for (FunctionValues v : vals) {
              if (v.boolVal(doc) != and) return !and;
            }
            return and;
          }
        };
      }
      case "gt" ->
          new ComparisonBoolFunction(vs(a.get(0)), vs(a.get(1)), "gt") {
            @Override
            public boolean compare(int doc, FunctionValues lhs, FunctionValues rhs) throws IOException {
              return lhs.doubleVal(doc) > rhs.doubleVal(doc);
            }
          };
      case "sqrt" ->
          new SimpleFloatFunction(vs(a.get(0))) {
            @Override
            protected String name() {
              return "sqrt";
            }

            @Override
            protected float func(int doc, FunctionValues vals) throws IOException {
              return (float) Math.sqrt(vals.floatVal(doc));
            }
          };
      case "docfreq" -> new DocFreqValueSource(a.get(0), a.get(1), a.get(0), new BytesRef(a.get(1)));
      case "idf" -> new IDFValueSource(a.get(0), a.get(1), a.get(0), new BytesRef(a.get(1)));
      case "termfreq" -> new TermFreqValueSource(a.get(0), a.get(1), a.get(0), new BytesRef(a.get(1)));
      case "tf" -> new TFValueSource(a.get(0), a.get(1), a.get(0), new BytesRef(a.get(1)));
      case "ttf" -> new TotalTermFreqValueSource(a.get(0), a.get(1), a.get(0), new BytesRef(a.get(1)));
      case "sttf" -> new SumTotalTermFreqValueSource(a.get(0));
      case "numdocs" -> new NumDocsValueSource();
      case "maxdoc" -> new MaxDocValueSource();
      case "norm" -> new NormValueSource(a.get(0));
      case "query" -> new QueryValueSource(query(a.get(0)), Float.parseFloat(a.get(1)));
      case "fvec" -> new FloatKnnVectorFieldSource(a.get(0));
      case "bvec" -> new ByteKnnVectorFieldSource(a.get(0));
      case "cfvec" -> {
        float[] v = new float[a.size()];
        for (int i = 0; i < v.length; i++) v[i] = Float.parseFloat(a.get(i));
        yield new ConstKnnFloatValueSource(v);
      }
      case "cbvec" -> {
        byte[] v = new byte[a.size()];
        for (int i = 0; i < v.length; i++) v[i] = Byte.parseByte(a.get(i));
        yield new ConstKnnByteVectorValueSource(v);
      }
      case "fsim" ->
          new FloatVectorSimilarityFunction(VectorSimilarityFunction.valueOf(a.get(0)), vs(a.get(1)), vs(a.get(2)));
      case "bsim" ->
          new ByteVectorSimilarityFunction(VectorSimilarityFunction.valueOf(a.get(0)), vs(a.get(1)), vs(a.get(2)));
      case "vector" -> new VectorValueSource(Arrays.asList(vsList(a)));
      case "fromdvs" -> ValueSource.fromDoubleValuesSource(dvs(a.get(0)));
      default -> throw new IllegalArgumentException(spec);
    };
  }

  /** A DoubleValuesSource: {@code vs(...)}, {@code lvs(...)}, {@code dint(f)} ... */
  static DoubleValuesSource dvs(String spec) {
    List<String> a = args(spec);
    return switch (name(spec)) {
      case "vs" -> vs(a.get(0)).asDoubleValuesSource();
      case "lvs" -> vs(a.get(0)).asLongValuesSource().toDoubleValuesSource();
      case "dint" -> DoubleValuesSource.fromIntField(a.get(0));
      case "dlong" -> DoubleValuesSource.fromLongField(a.get(0));
      case "dfloat" -> DoubleValuesSource.fromFloatField(a.get(0));
      case "ddouble" -> DoubleValuesSource.fromDoubleField(a.get(0));
      case "dconst" -> DoubleValuesSource.constant(Double.parseDouble(a.get(0)));
      case "scores" -> DoubleValuesSource.SCORES;
      case "dquery" -> DoubleValuesSource.fromQuery(query(a.get(0)));
      case "irdocfreq" -> IndexReaderFunctions.docFreq(new Term(a.get(0), a.get(1)));
      case "irmaxdoc" -> IndexReaderFunctions.maxDoc();
      case "irnumdocs" -> IndexReaderFunctions.numDocs();
      case "irnumdeleted" -> IndexReaderFunctions.numDeletedDocs();
      case "irsttf" -> IndexReaderFunctions.sumTotalTermFreq(a.get(0)).toDoubleValuesSource();
      case "irtermfreq" -> IndexReaderFunctions.termFreq(new Term(a.get(0), a.get(1)));
      case "irttf" -> IndexReaderFunctions.totalTermFreq(new Term(a.get(0), a.get(1)));
      case "irsumdocfreq" -> IndexReaderFunctions.sumDocFreq(a.get(0));
      case "irdoccount" -> IndexReaderFunctions.docCount(a.get(0));
      default -> throw new IllegalArgumentException(spec);
    };
  }

  /** A predicate: {@code gt:x}, {@code le:x}, {@code nan}. */
  static DoublePredicate predicate(String spec) {
    String[] p = spec.split(":");
    return switch (p[0]) {
      case "gt" -> v -> v > Double.parseDouble(p[1]);
      case "le" -> v -> v <= Double.parseDouble(p[1]);
      case "nan" -> Double::isNaN;
      default -> throw new IllegalArgumentException(spec);
    };
  }

  // ---------------------------------------------------------------------------------------------
  // Printing
  // ---------------------------------------------------------------------------------------------

  static String hex(float f) {
    return Integer.toHexString(Float.floatToIntBits(f));
  }

  static String hex(double d) {
    return Long.toHexString(Double.doubleToLongBits(d));
  }

  /** The exception's name as the Rust test maps its errors. */
  static String err(Throwable e) {
    if (e instanceof NumberFormatException) return "!IllegalArgumentException";
    if (e instanceof NullPointerException) return "!IllegalStateException";
    return "!" + e.getClass().getSimpleName();
  }

  static String clean(String s) {
    return s == null ? "null" : s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n");
  }

  interface Get {
    String get() throws Exception;
  }

  static String g(Get f) {
    try {
      return clean(f.get());
    } catch (Exception | AssertionError e) {
      return err(e);
    }
  }

  static String object(Object o) {
    if (o == null) return "null";
    String t =
        switch (o) {
          case Float x -> "F";
          case Double x -> "D";
          case Integer x -> "I";
          case Long x -> "L";
          case String x -> "S";
          case Boolean x -> "B";
          default -> o.getClass().getSimpleName();
        };
    if (o instanceof Float f) return t + hex(f);
    if (o instanceof Double d) return t + hex(d);
    return t + o;
  }

  static String fill(MutableValue m) {
    String kind = m.getClass().getSimpleName().replace("MutableValue", "");
    return kind + ":" + m.exists() + ":" + m.toString();
  }

  // ---------------------------------------------------------------------------------------------
  // values.tsv
  // ---------------------------------------------------------------------------------------------

  static final String[] VALUE_SPECS = {
    "int(i)", "long(l)", "float(f)", "double(d)", "int(l)", "float(i)", "long(nosuch)",
    "mint(mi,MIN)", "mint(mi,MAX)", "mlong(ml,MIN)", "mlong(ml,MAX)", "mfloat(mf,MIN)",
    "mfloat(mf,MAX)", "mdouble(md,MIN)", "mdouble(md,MAX)",
    "bytes(s)", "bytes(b)", "bytes(nosuch)", "sset(ss,MIN)", "sset(ss,MAX)", "sset(ss,MIDDLE_MIN)",
    "sset(ss,MIDDLE_MAX)", "enum(e)", "joindf(k,body)",
    "const(1.5)", "const(-2.0)", "dconst(2.25)", "dconst(1.0E10)", "literal(hello)",
    "sum(int(i),float(f),const(0.5))", "product(float(f),double(d))", "max(int(i),float(f))",
    "min(int(i),float(f),long(l))", "max(long(nosuch),int(i))", "div(float(f),int(i))",
    "pow(float(f),const(0.5))", "pow(int(i),const(2.5))", "linear(float(f),2.5,-1.0)",
    "recip(int(i),0.5,3.0,2.0)", "map(int(i),0.0,10.0,const(100.0),null)",
    "map(float(f),-1.0,1.0,int(i),const(-7.0))", "scale(float(f),0.0,1.0)",
    "scale(int(i),-1.0,1.0)", "scale(query(t:red,0.0),0.0,10.0)", "def(int(i),float(f),const(9.0))",
    "if(int(i),float(f),const(3.0))", "if(exists(float(f)),float(f),const(-1.0))",
    "exists(int(i))", "and(int(i),float(f))", "or(int(i),long(l))", "gt(int(i),float(f))",
    "sqrt(float(f))", "sum(mint(mi,MAX),mdouble(md,MIN))",
    "docfreq(body,red)", "docfreq(body,nosuch)", "idf(body,red)", "idf(body,old)",
    "termfreq(body,red)", "termfreq(body,nosuch)", "termfreq(id,d3)", "tf(body,red)",
    "tf(body,blue)", "ttf(body,red)", "sttf(body)", "sttf(nosuch)", "numdocs()", "maxdoc()",
    "norm(body)", "norm(nosuch)", "query(t:red,0.0)", "query(or:blue:green,-1.0)",
    "query(and:red:blue,2.5)", "sum(query(t:fast,0.0),int(i))",
    "fvec(fv)", "bvec(bv)", "fvec(bv)", "fvec(i)", "cfvec(0.5,-1.0,2.0,0.25)", "cbvec(1,-2,3,100)",
    "fsim(EUCLIDEAN,fvec(fv),cfvec(0.5,-1.0,2.0,0.25))", "fsim(DOT_PRODUCT,fvec(fv),fvec(fv))",
    "fsim(COSINE,fvec(fv),cfvec(1.0,1.0,1.0,1.0))",
    "fsim(MAXIMUM_INNER_PRODUCT,fvec(fv),cfvec(0.5,0.5,0.5,0.5))",
    "bsim(EUCLIDEAN,bvec(bv),cbvec(1,-2,3,100))", "bsim(DOT_PRODUCT,bvec(bv),cbvec(1,-2,3,100))",
    "bsim(COSINE,bvec(bv),cbvec(1,-2,3,100))", "vector(int(i),float(f))",
    "vector(int(i),float(f),long(l))", "fromdvs(dint(i))", "fromdvs(ddouble(d))",
    "fromdvs(vs(float(f)))", "fromdvs(dquery(t:red))", "fromdvs(irmaxdoc())",
  };

  static final String[] BM25_SPECS = {"idf(body,red)", "tf(body,red)", "norm(body)"};

  void values(StringBuilder out) throws IOException {
    for (String spec : VALUE_SPECS) dump(out, spec, classic, "classic");
    for (String spec : BM25_SPECS) dump(out, spec, bm25, "bm25");
  }

  void dump(StringBuilder out, String spec, IndexSearcher searcher, String sim) throws IOException {
    ValueSource vs = vs(spec);
    Map<Object, Object> context = ValueSource.newContext(searcher);
    String head = sim + "\t" + spec;
    out.append(head).append("\tdesc\t").append(clean(vs.description())).append('\n');
    try {
      vs.createWeight(context, searcher);
    } catch (Exception e) {
      out.append(head).append("\tweight\t").append(err(e)).append('\n');
      return;
    }
    List<LeafReaderContext> leaves = reader.leaves();
    for (int seg = 0; seg < leaves.size(); seg++) {
      LeafReaderContext leaf = leaves.get(seg);
      FunctionValues fv;
      FunctionValues filler;
      try {
        fv = vs.getValues(context, leaf);
        filler = vs.getValues(context, leaf);
      } catch (Exception e) {
        out.append(head).append('\t').append(seg).append("\t").append(err(e)).append('\n');
        continue;
      }
      FunctionValues.ValueFiller f = filler.getValueFiller();
      boolean multi = vs instanceof VectorValueSource;
      for (int doc = 0; doc < leaf.reader().maxDoc(); doc++) {
        final int d = doc;
        List<String> cols = new ArrayList<>();
        cols.add(g(() -> String.valueOf(fv.exists(d))));
        if (multi) {
          int n = ((VectorValueSource) vs).dimension();
          cols.add(g(() -> { float[] v = new float[n]; fv.floatVal(d, v); StringBuilder b = new StringBuilder(); for (float x : v) b.append(hex(x)).append(' '); return b.toString().trim(); }));
          cols.add(g(() -> { double[] v = new double[n]; fv.doubleVal(d, v); StringBuilder b = new StringBuilder(); for (double x : v) b.append(hex(x)).append(' '); return b.toString().trim(); }));
          cols.add(g(() -> { int[] v = new int[n]; fv.intVal(d, v); return Arrays.toString(v); }));
          cols.add(g(() -> { long[] v = new long[n]; fv.longVal(d, v); return Arrays.toString(v); }));
          cols.add(g(() -> { byte[] v = new byte[n]; fv.byteVal(d, v); return Arrays.toString(v); }));
          cols.add(g(() -> { short[] v = new short[n]; fv.shortVal(d, v); return Arrays.toString(v); }));
          cols.add(g(() -> { String[] v = new String[n]; fv.strVal(d, v); return Arrays.toString(v); }));
          cols.add(g(() -> fv.toString(d)));
        } else {
          cols.add(g(() -> String.valueOf(fv.byteVal(d))));
          cols.add(g(() -> String.valueOf(fv.shortVal(d))));
          cols.add(g(() -> hex(fv.floatVal(d))));
          cols.add(g(() -> hex(fv.doubleVal(d))));
          cols.add(g(() -> String.valueOf(fv.intVal(d))));
          cols.add(g(() -> String.valueOf(fv.longVal(d))));
          cols.add(g(() -> String.valueOf(fv.boolVal(d))));
          cols.add(g(() -> fv.strVal(d)));
          cols.add(g(() -> {
            BytesRefBuilder b = new BytesRefBuilder();
            boolean has = fv.bytesVal(d, b);
            return has + ":" + b.get().utf8ToString();
          }));
          cols.add(g(() -> object(fv.objectVal(d))));
          cols.add(g(() -> String.valueOf(fv.ordVal(d))));
          cols.add(g(() -> {
            float[] v = fv.floatVectorVal(d);
            return v == null ? "null" : Arrays.toString(v);
          }));
          cols.add(g(() -> {
            byte[] v = fv.byteVectorVal(d);
            return v == null ? "null" : Arrays.toString(v);
          }));
          cols.add(g(() -> fv.toString(d)));
          cols.add(g(() -> {
            f.fillValue(d);
            return fill(f.getValue());
          }));
        }
        out.append(head).append('\t').append(seg).append('\t').append(doc);
        for (String c : cols) out.append('\t').append(c);
        out.append('\n');
      }
      final FunctionValues ff = fv;
      out.append(head).append('\t').append(seg).append("\tnumOrd\t").append(g(() -> String.valueOf(ff.numOrd()))).append('\n');
    }
  }

  // ---------------------------------------------------------------------------------------------
  // searches.tsv
  // ---------------------------------------------------------------------------------------------

  static final int[] EXPLAIN = {0, 7, 29, 61};

  void search(StringBuilder out, String kind, String spec, IndexSearcher searcher, Query q) {
    searchHits(out, kind, spec, searcher, q);
    String head = kind + "\t" + spec;
    for (int doc : EXPLAIN) {
      String e = g(() -> searcher.explain(q, doc).toString());
      out.append(head).append("\texplain ").append(doc).append('\t').append(e).append('\n');
    }
  }

  void searchHits(StringBuilder out, String kind, String spec, IndexSearcher searcher, Query q) {
    String head = kind + "\t" + spec;
    try {
      TopDocs td = searcher.search(q, 1000);
      StringBuilder b = new StringBuilder();
      b.append(td.totalHits.value()).append(' ');
      for (ScoreDoc sd : td.scoreDocs) b.append(sd.doc).append(':').append(hex(sd.score)).append(',');
      out.append(head).append("\thits\t").append(b).append('\n');
    } catch (Exception e) {
      out.append(head).append("\thits\t").append(err(e)).append('\n');
    }
  }

  static final String[] FQ_SPECS = {
    "int(i)", "float(f)", "double(d)", "sum(int(i),float(f),const(0.5))", "div(float(f),int(i))",
    "linear(float(f),2.5,-1.0)", "max(int(i),float(f))", "scale(float(f),0.0,1.0)",
    "query(t:red,0.5)", "termfreq(body,red)", "docfreq(body,blue)", "pow(int(i),const(2.5))",
    "mdouble(md,MAX)", "fsim(EUCLIDEAN,fvec(fv),cfvec(0.5,-1.0,2.0,0.25))", "bytes(s)",
    "fromdvs(dint(i))", "joindf(k,body)", "map(float(f),-1.0,1.0,int(i),const(-7.0))",
  };

  static final String[][] RANGE_SPECS = {
    {"int(i)", "3", "20", "true", "true"}, {"int(i)", "3", "20", "false", "false"},
    {"int(i)", null, "0", "true", "true"}, {"long(l)", "-100", "500", "true", "false"},
    {"long(l)", "x", "500", "true", "false"}, {"float(f)", "-1.5", "6", "true", "true"},
    {"float(f)", null, null, "true", "true"}, {"double(d)", "0", "30.5", "false", "true"},
    {"enum(e)", "one", "three", "true", "true"}, {"enum(e)", "1", "nosuch", "false", "true"},
    {"bytes(s)", "blue0", "red1", "true", "false"}, {"sset(ss,MAX)", "fast", null, "false", true + ""},
    {"sum(int(i),float(f))", "0", "10", "true", "true"}, {"exists(int(i))", "0.5", "2", "true", "true"},
    {"termfreq(body,red)", "1", null, "true", "true"}, {"query(t:red,0.0)", "0.1", null, "true", "true"},
    {"literal(x)", "0", "1", "true", "true"}, {"mint(mi,MAX)", "5", "30", "true", "true"},
  };

  static final String[][] FSQ_SPECS = {
    {"t:red", "plain", "dint(i)"}, {"t:red", "plain", "vs(float(f))"},
    {"or:blue:green", "plain", "vs(sum(int(i),float(f)))"}, {"all", "plain", "ddouble(d)"},
    {"t:red", "plain", "scores"}, {"or:red:old", "boostval", "dint(i)"},
    {"or:red:old", "boostval", "vs(float(f))"}, {"t:blue", "boostq", "t:red:2.5"},
    {"or:red:fast", "boostq", "and:blue:green:0.5"}, {"all", "plain", "irdocfreq(body,red)"},
    {"all", "plain", "irmaxdoc()"}, {"all", "plain", "irnumdocs()"}, {"all", "plain", "irnumdeleted()"},
    {"all", "plain", "irsttf(body)"}, {"t:red", "plain", "irtermfreq(body,red)"},
    {"all", "plain", "irttf(body,red)"}, {"all", "plain", "irsumdocfreq(body)"},
    {"all", "plain", "irdoccount(body)"}, {"all", "plain", "lvs(int(i))"},
    {"t:big", "plain", "dquery(t:small)"}, {"all", "plain", "vs(docfreq(body,red))"},
    {"all", "plain", "vs(ttf(body,red))"}, {"all", "plain", "vs(scale(int(i),0.0,1.0))"},
    {"t:red", "boostval", "vs(query(t:blue,1.0))"},
  };

  static final String[][] FMQ_SPECS = {
    {"dint(i)", "gt:10"}, {"ddouble(d)", "le:0"}, {"vs(float(f))", "nan"}, {"vs(int(i))", "gt:-100"},
    {"dfloat(f)", "gt:3.5"}, {"irmaxdoc()", "gt:5"},
  };

  void searches(StringBuilder out) throws IOException {
    for (String spec : FQ_SPECS) {
      search(out, "fq", spec, bm25, new FunctionQuery(vs(spec)));
      search(out, "fqboost", spec, bm25, new BoostQuery(new FunctionQuery(vs(spec)), 2.5f));
      BooleanQuery.Builder b = new BooleanQuery.Builder();
      b.add(new TermQuery(new Term("body", "red")), BooleanClause.Occur.MUST);
      b.add(new FunctionQuery(vs(spec)), BooleanClause.Occur.SHOULD);
      search(out, "fqbool", spec, bm25, b.build());
    }
    for (String spec : new String[] {"tf(body,red)", "idf(body,red)", "norm(body)"}) {
      search(out, "fqclassic", spec, classic, new FunctionQuery(vs(spec)));
      search(out, "fq", spec, bm25, new FunctionQuery(vs(spec)));
    }
    for (String[] r : RANGE_SPECS) {
      String spec = r[0] + "|" + r[1] + "|" + r[2] + "|" + r[3] + "|" + r[4];
      Query q =
          new FunctionRangeQuery(
              vs(r[0]), r[1], r[2], Boolean.parseBoolean(r[3]), Boolean.parseBoolean(r[4]));
      search(out, "frange", spec, bm25, q);
      BooleanQuery.Builder b = new BooleanQuery.Builder();
      b.add(new TermQuery(new Term("body", "blue")), BooleanClause.Occur.MUST);
      b.add(q, BooleanClause.Occur.FILTER);
      search(out, "frangefilter", spec, bm25, b.build());
    }
    for (String[] f : FSQ_SPECS) {
      String spec = f[0] + "|" + f[1] + "|" + f[2];
      Query in = query(f[0]);
      Query q =
          switch (f[1]) {
            case "plain" -> new FunctionScoreQuery(in, dvs(f[2]));
            case "boostval" -> FunctionScoreQuery.boostByValue(in, dvs(f[2]));
            default -> {
              int last = f[2].lastIndexOf(':');
              yield FunctionScoreQuery.boostByQuery(
                  in, query(f[2].substring(0, last)), Float.parseFloat(f[2].substring(last + 1)));
            }
          };
      search(out, "fsq", spec, bm25, q);
      search(out, "fsqboost", spec, bm25, new BoostQuery(q, 3.0f));
      BooleanQuery.Builder b = new BooleanQuery.Builder();
      b.add(q, BooleanClause.Occur.SHOULD);
      b.add(new TermQuery(new Term("body", "slow")), BooleanClause.Occur.SHOULD);
      search(out, "fsqbool", spec, bm25, b.build());
      BooleanQuery.Builder fb = new BooleanQuery.Builder();
      fb.add(new TermQuery(new Term("body", "big")), BooleanClause.Occur.MUST);
      fb.add(q, BooleanClause.Occur.FILTER);
      // Hits only: a filter's term explains itself unscored ("with freq of"), which the
      // port's explain does not render.
      searchHits(out, "fsqfilter", spec, bm25, fb.build());
    }
    for (String[] f : FMQ_SPECS) {
      String spec = f[0] + "|" + f[1];
      Query q = new FunctionMatchQuery(dvs(f[0]), predicate(f[1]));
      search(out, "fmq", spec, bm25, q);
      search(out, "fmqboost", spec, bm25, new BoostQuery(q, 0.5f));
    }
    for (String spec : new String[] {"sum(int(i),float(f))", "query(t:red,0.0)", "int(i)", "double(d)", "scale(float(f),0.0,1.0)"}) {
      for (boolean reverse : new boolean[] {false, true}) {
        String head = "sort\t" + spec + "|" + reverse;
        try {
          Sort sort = new Sort(vs(spec).getSortField(reverse), SortField.FIELD_DOC);
          TopDocs td = bm25.search(new MatchAllDocsQuery(), 12, sort.rewrite(bm25));
          StringBuilder b = new StringBuilder();
          for (ScoreDoc sd : td.scoreDocs) b.append(sd.doc).append(',');
          out.append(head).append("\thits\t").append(b).append('\n');
        } catch (Exception e) {
          out.append(head).append("\thits\t").append(err(e)).append('\n');
        }
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // groups.tsv: GroupingSearch(ValueSource, context), lucene-grouping's ValueSourceGroupSelector
  // ---------------------------------------------------------------------------------------------

  static final String[] GROUP_SPECS = {
    "int(i)", "long(l)", "float(f)", "double(d)", "bytes(s)", "bytes(b)", "sset(ss,MAX)", "enum(e)",
    "query(t:red,0.0)", "scale(int(i),0.0,1.0)", "def(int(i),const(-1.0))", "exists(int(i))",
    "mint(mi,MIN)", "joindf(k,body)",
  };

  static String group(Object v) {
    if (v == null) return "null";
    MutableValue m = (MutableValue) v;
    return m.exists() ? fill(m) : "missing";
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  void groups(StringBuilder out) {
    for (String spec : GROUP_SPECS) {
      for (String qs : new String[] {"all", "t:red"}) {
        String head = spec + "\t" + qs;
        try {
          ValueSource vs = vs(spec);
          Map<Object, Object> context = ValueSource.newContext(bm25);
          vs.createWeight(context, bm25);
          GroupingSearch gs = new GroupingSearch(vs, context);
          gs.setGroupDocsLimit(2);
          gs.setAllGroups(true);
          TopGroups<?> tg = gs.search(bm25, query(qs), 0, 20);
          StringBuilder b = new StringBuilder();
          b.append(tg.totalHitCount).append(' ').append(tg.totalGroupedHitCount).append(' ');
          b.append(tg.totalGroupCount).append(' ').append(gs.getAllMatchingGroups().size());
          for (GroupDocs<?> g : tg.groups) {
            b.append(" | ").append(group(g.groupValue())).append(" =");
            for (ScoreDoc sd : g.scoreDocs()) b.append(' ').append(sd.doc).append(':').append(hex(sd.score));
          }
          out.append(head).append('\t').append(clean(b.toString())).append('\n');
        } catch (Exception e) {
          out.append(head).append('\t').append(err(e)).append('\n');
        }
      }
    }
  }
}
