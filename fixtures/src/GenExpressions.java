import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Comparator;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleDocValuesField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.expressions.Expression;
import org.apache.lucene.search.Rescorer;
import org.apache.lucene.expressions.SimpleBindings;
import org.apache.lucene.expressions.js.JavascriptCompiler;
import org.apache.lucene.expressions.js.VariableContext;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * M14 T14.2 ({@code expressions}): what {@code JavascriptCompiler} makes of every expression of
 * {@code fixtures/corpus/expressions.txt} (the expressions of Lucene's own expression tests) and of
 * 4000 random ones, 1500 random token soups and 65 long runs of one precedence level's operators
 * (a fixed seed), compiled with the default functions ({@code D}) and with two
 * custom ones added ({@code C}); and what {@code VariableContext.parse} makes of every variable.
 *
 * <p>{@code expressions.tsv}: {@code E}, the function set, the expression, then {@code OK}, the
 * variables (joined by U+001F) and the value under each of four variable assignments ({@code
 * Double.doubleToLongBits} in hex: NaN canonical) -- or {@code ERR}, the exception's simple class
 * name, the {@code ParseException} offset ({@code -1} for any other) and the message. {@code V}
 * rows: a variable, then its contexts or {@code ERR}. {@code
 * crates/lucene-search/tests/expressions_fixtures.rs} compares.
 *
 * <p>{@code index/} is a two-segment index (the first with deletions) with sparse {@code n}
 * (long) and {@code d} (double) doc values and a {@code body} text field; {@code index.tsv} holds,
 * for expressions over {@code n}, {@code d}, {@code _score} and a nested expression: {@code X}
 * rows, every document's value; {@code S} rows, the top hits of a term query sorted by the
 * expression (doc and sort value bits); {@code R} rows, {@code ExpressionRescorer}'s rescoring of
 * that query's hits (doc and score bits); {@code P} rows, its explanation of three of them.
 * {@code crates/lucene-search/tests/expressions_index.rs} compares.
 */
public class GenExpressions {

  static String esc(String s) {
    return AnalysisRows.esc(s);
  }

  static final double[][] SETS = {
    {0, 1, -1, 2.5, 3},
    {Double.NaN, Double.POSITIVE_INFINITY, -0.0, 1e20, Double.NEGATIVE_INFINITY},
    {3, 7.9, -7.9, 1L << 40, 0.5},
    {-2.5e-300, 9.223372036854775807E18, -9.3e18, 2147483648.0, -2147483649.5},
  };

  public static double foo() {
    return 5;
  }

  public static double bar(double x) {
    return x * 3 + 1;
  }

  public static double baz(double a, double b, double c) {
    return a * 100 + b * 10 + c;
  }

  static final String[] FUNCS = {
    "abs:1", "acos:1", "acosh:1", "asin:1", "asinh:1", "atan:1", "atan2:2", "atanh:1", "ceil:1",
    "cos:1", "cosh:1", "exp:1", "floor:1", "haversin:4", "haversinMeters:4", "ln:1", "log10:1",
    "logn:2", "max:2", "min:2", "pow:2", "sin:1", "sinh:1", "sqrt:1", "tan:1", "tanh:1"
  };
  static final String[] BINARY = {
    "*", "/", "%", "+", "-", "<<", ">>", ">>>", "<", "<=", ">", ">=", "==", "!=", "&", "^", "|",
    "&&", "||"
  };
  static final String[] UNARY = {"!", "~", "+", "-"};
  static final String[] VARS = {
    "a", "b", "_c", "$d", "x.y", "doc['f'].value", "doc[\"g\"].value", "arr[0]", "m.n()",
    "o[1][2].p", "s['it\\'s']", "q[\"a\\\\b\"]"
  };
  static final String[] LITERALS = {
    "0", "1", "2", "3", "7", "10", "0.5", ".5", "5.", "1e3", "1.5E-2", "1e300", "2147483647",
    "2147483648", "4294967296", "9223372036854775807", "0x7fffffffffffffff", "0xff", "0X1F",
    "017", "00", "0777777777777777777777", "123456789012345678901234567890", "1e-320", "3.75"
  };

  static String randomExpr(Random r, int depth) {
    int pick = depth <= 0 ? r.nextInt(2) : r.nextInt(9);
    switch (pick) {
      case 0:
        return LITERALS[r.nextInt(LITERALS.length)];
      case 1:
        return VARS[r.nextInt(VARS.length)];
      case 2:
      case 3:
      case 4:
        return randomExpr(r, depth - 1)
            + " " + BINARY[r.nextInt(BINARY.length)] + " "
            + randomExpr(r, depth - 1);
      case 5:
        return UNARY[r.nextInt(UNARY.length)] + randomExpr(r, depth - 1);
      case 6:
        return "(" + randomExpr(r, depth - 1) + ")";
      case 7:
        return randomExpr(r, depth - 1)
            + " ? " + randomExpr(r, depth - 1)
            + " : " + randomExpr(r, depth - 1);
      default:
        {
          String[] f = FUNCS[r.nextInt(FUNCS.length)].split(":");
          int arity = Integer.parseInt(f[1]);
          if (r.nextInt(20) == 0) arity = r.nextInt(5);
          StringBuilder b = new StringBuilder(f[0]).append('(');
          for (int i = 0; i < arity; i++) {
            if (i > 0) b.append(", ");
            b.append(randomExpr(r, depth - 1));
          }
          return b.append(')').toString();
        }
    }
  }

  /** The binary operators by precedence level, for chains of one level. */
  static final String[][] LEVELS = {
    {"*", "/", "%"}, {"+", "-"}, {"<<", ">>", ">>>"}, {"<", "<=", ">", ">="}, {"==", "!="},
    {"&"}, {"^"}, {"|"}, {"&&"}, {"||"}
  };

  /**
   * A flat run of {@code n} operators of one precedence level over operands at most {@code depth}
   * deep (at depth 0, literals and variables only: no call that might fail its arity).
   */
  static String randomChain(Random r, String[] level, int n, int depth) {
    StringBuilder b = new StringBuilder(randomExpr(r, r.nextInt(depth + 1)));
    for (int i = 0; i < n; i++) {
      b.append(' ').append(level[r.nextInt(level.length)]).append(' ');
      b.append(randomExpr(r, r.nextInt(depth + 1)));
    }
    return b.toString();
  }

  static final String[] SOUP = {
    "(", ")", ",", "!", "~", "*", "/", "%", "+", "-", "<<", ">>", ">>>", "<", "<=", ">", ">=",
    "==", "!=", "&", "^", "|", "&&", "||", "?", ":", " ", "a", "b.c", "1", "0x", "08", "1e", "@",
    "#", "[", "]", "'", "\"", "sqrt", "max", "0.", "x[", "x['a", "\n", "=", "foo", "07", "é"
  };

  static String randomSoup(Random r) {
    StringBuilder b = new StringBuilder();
    int n = 1 + r.nextInt(6);
    for (int i = 0; i < n; i++) b.append(SOUP[r.nextInt(SOUP.length)]);
    return b.toString();
  }

  static final class Fixed extends DoubleValues {
    final double v;

    Fixed(double v) {
      this.v = v;
    }

    @Override
    public double doubleValue() {
      return v;
    }

    @Override
    public boolean advanceExact(int doc) {
      return true;
    }
  }

  static String hex(double d) {
    return Long.toHexString(Double.doubleToLongBits(d));
  }

  static void compileRow(
      StringBuilder rows, String mode, String text, Map<String, MethodHandle> fns, Set<String> vars) {
    rows.append("E\t").append(mode).append('\t').append(esc(text)).append('\t');
    try {
      Expression e = JavascriptCompiler.compile(text, fns);
      rows.append("OK\t").append(esc(String.join("\u001f", e.variables))).append('\t');
      for (int k = 0; k < SETS.length; k++) {
        DoubleValues[] values = new DoubleValues[e.variables.length];
        for (int i = 0; i < values.length; i++) {
          values[i] = new Fixed(SETS[k][(i + k) % SETS[k].length]);
        }
        if (k > 0) rows.append(',');
        rows.append(hex(e.evaluate(values)));
      }
      for (String v : e.variables) vars.add(v);
    } catch (ParseException pe) {
      rows.append("ERR\tParseException\t").append(pe.getErrorOffset()).append('\t')
          .append(esc(pe.getMessage()));
    } catch (Exception ex) {
      rows.append("ERR\t").append(ex.getClass().getSimpleName()).append("\t-1\t")
          .append(esc(String.valueOf(ex.getMessage())));
    }
    rows.append('\n');
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("expressions");
    Files.createDirectories(out);

    Map<String, MethodHandle> custom = new HashMap<>(JavascriptCompiler.DEFAULT_FUNCTIONS);
    MethodHandles.Lookup lookup = MethodHandles.publicLookup();
    custom.put("foo", lookup.findStatic(GenExpressions.class, "foo", MethodType.methodType(double.class)));
    custom.put("bar", lookup.findStatic(GenExpressions.class, "bar", MethodType.methodType(double.class, double.class)));
    custom.put(
        "method.method",
        lookup.findStatic(
            GenExpressions.class,
            "baz",
            MethodType.methodType(double.class, double.class, double.class, double.class)));

    List<String> inputs = new ArrayList<>(AnalysisRows.corpus("expressions.txt"));
    Random r = new Random(0x5eed1234L);
    for (int i = 0; i < 4000; i++) inputs.add(randomExpr(r, 1 + r.nextInt(5)));
    for (int i = 0; i < 1500; i++) inputs.add(randomSoup(r));
    // chains longer than the Rust port's nesting bound, which it folds flat
    for (int i = 0; i < 60; i++) {
      inputs.add(randomChain(r, LEVELS[i % LEVELS.length], 17 + r.nextInt(100), 1));
    }
    for (String[] level : new String[][] {{"+", "-"}, {"|"}, {"&&"}, {"*", "/"}, {"<", ">="}}) {
      inputs.add(randomChain(r, level, 400, 0));
    }

    StringBuilder rows = new StringBuilder();
    Set<String> vars = new LinkedHashSet<>();
    for (String text : inputs) {
      compileRow(rows, "D", text, JavascriptCompiler.DEFAULT_FUNCTIONS, vars);
      compileRow(rows, "C", text, custom, vars);
    }
    for (String extra :
        new String[] {"x[2147483648]", "x[99999999999]", "a", "a.b.c()", "a()", "x['']"}) {
      vars.add(extra);
    }
    for (String v : vars) {
      rows.append("V\t").append(esc(v)).append('\t');
      try {
        VariableContext[] cs = VariableContext.parse(v);
        for (int i = 0; i < cs.length; i++) {
          if (i > 0) rows.append('\u001f');
          rows.append(cs[i].type).append(':').append(esc(String.valueOf(cs[i].text)))
              .append(':').append(cs[i].integer);
        }
      } catch (Exception ex) {
        rows.append("ERR ").append(ex.getClass().getSimpleName());
      }
      rows.append('\n');
    }
    Files.writeString(out.resolve("expressions.tsv"), rows.toString(), StandardCharsets.UTF_8);
    indexRows(out);
  }

  static final String[] INDEX_EXPRESSIONS = {
    "n * 2 + d", "inner - n + sqrt(abs(d))", "_score * 2 + n", "ln(1 + abs(d)) * inner + _score",
    "n > 100 ? d : -n",
  };

  static void indexRows(Path out) throws Exception {
    Path dirPath = out.resolve("index");
    if (Files.exists(dirPath)) {
      try (Stream<Path> st = Files.walk(dirPath)) {
        st.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(dirPath);
    Random r = new Random(0x1dec5L);
    StringBuilder rows = new StringBuilder();
    try (Directory dir = FSDirectory.open(dirPath)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int id = 0;
        for (int size : new int[] {30, 20}) {
          for (int i = 0; i < size; i++, id++) {
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            StringBuilder body = new StringBuilder();
            for (int t = 0, len = 2 + r.nextInt(6); t < len; t++) {
              body.append(" w").append(r.nextInt(4));
            }
            doc.add(new TextField("body", body.toString(), Field.Store.NO));
            if (id % 5 != 0) doc.add(new NumericDocValuesField("n", r.nextInt(401) - 200));
            if (id % 3 != 2) doc.add(new DoubleDocValuesField("d", r.nextDouble() * 50 - 25));
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int d = 0; d < 30; d += 7) w.deleteDocuments(new Term("id", Integer.toString(d)));
        w.commit();
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        SimpleBindings b = new SimpleBindings();
        b.add("n", DoubleValuesSource.fromLongField("n"));
        b.add("d", DoubleValuesSource.fromDoubleField("d"));
        b.add("_score", DoubleValuesSource.SCORES);
        b.add("inner", JavascriptCompiler.compile("n * 2 + d"));
        Query q = new TermQuery(new Term("body", "w1"));
        for (String text : INDEX_EXPRESSIONS) {
          Expression e = JavascriptCompiler.compile(text);
          DoubleValuesSource src = e.getDoubleValuesSource(b);
          if (!src.needsScores()) {
            for (LeafReaderContext leaf : reader.leaves()) {
              DoubleValues v = src.getValues(leaf, null);
              for (int doc = 0; doc < leaf.reader().maxDoc(); doc++) {
                rows.append("X\t").append(text).append('\t').append(leaf.docBase + doc).append('\t');
                rows.append(v.advanceExact(doc) ? Long.toHexString(Double.doubleToRawLongBits(v.doubleValue())) : "-");
                rows.append('\n');
              }
            }
          }
          TopDocs sorted = searcher.search(q, 12, new Sort(e.getSortField(b, true)));
          rows.append("S\t").append(text).append('\t');
          for (ScoreDoc sd : sorted.scoreDocs) {
            FieldDoc fd = (FieldDoc) sd;
            rows.append(sd.doc).append(':')
                .append(Long.toHexString(Double.doubleToRawLongBits((Double) fd.fields[0]))).append(',');
          }
          rows.append('\n');
          TopDocs first = searcher.search(q, 40);
          Rescorer rescorer = e.getRescorer(b);
          TopDocs rescored = rescorer.rescore(searcher, first, 10);
          rows.append("R\t").append(text).append('\t');
          for (ScoreDoc sd : rescored.scoreDocs) {
            rows.append(sd.doc).append(':').append(Integer.toHexString(Float.floatToRawIntBits(sd.score))).append(',');
          }
          rows.append('\n');
          for (int k = 0; k < 3 && k < rescored.scoreDocs.length; k++) {
            int doc = rescored.scoreDocs[k].doc;
            String ex = rescorer.explain(searcher, searcher.explain(q, doc), doc).toString();
            rows.append("P\t").append(text).append('\t').append(doc).append('\t').append(esc(ex)).append('\n');
          }
        }
      }
    }
    Files.writeString(out.resolve("index.tsv"), rows.toString(), StandardCharsets.UTF_8);
  }
}
