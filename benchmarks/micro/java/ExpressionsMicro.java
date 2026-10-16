import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.expressions.Expression;
import org.apache.lucene.expressions.js.JavascriptCompiler;
import org.apache.lucene.search.DoubleValues;

/**
 * M14 T14.2's pairs (the Rust twin is {@code benchmarks/rust-runner/src/micro_expressions.rs}):
 * {@code expr_compile}, every expression of {@code fixtures/corpus/expressions.txt} that compiles
 * with the default functions (units: expressions); {@code expr_<shape>}, one expression evaluated
 * over 1,024 documents' values of {@code a}, {@code b} and {@code c} (units: evaluations). Output is
 * {@code name\tns_per_unit\tunits}.
 */
public class ExpressionsMicro {
  static final String[][] EVAL = {
    {"expr_arith", "a * 2 + b - c / 3"},
    {"expr_math", "sqrt(abs(b)) + ln(1 + abs(a)) * cos(c)"},
    {"expr_cond", "a > 100 ? b : (c < 0 && b != 0 ? -a : c)"},
    {"expr_bits", "((a << 2) | 3) ^ (c >>> 1)"},
    {"expr_haversin", "haversin(a / 10, b, c / 10, b + 1)"},
    {"expr_funcs", "max(a, b) * min(c, 1) + pow(b, 2) + floor(c)"},
  };
  static final int DOCS = 1024;

  /** The value of variable {@code k} for {@code doc}, as the Rust side computes it. */
  static double value(int doc, int k) {
    return (double) (((long) doc * 7919 + (long) k * 104729) % 401 - 200) + (doc % 7) / 8.0;
  }

  static final class ArrayValues extends DoubleValues {
    final double[] values;
    int doc;

    ArrayValues(double[] values) {
      this.values = values;
    }

    @Override
    public double doubleValue() {
      return values[doc];
    }

    @Override
    public boolean advanceExact(int doc) {
      this.doc = doc;
      return true;
    }
  }

  public static void main(String[] args) throws Exception {
    List<String> texts = new ArrayList<>();
    for (String line :
        Files.readAllLines(Path.of("fixtures/corpus/expressions.txt"), StandardCharsets.UTF_8)) {
      try {
        JavascriptCompiler.compile(line);
        texts.add(line);
      } catch (Exception | StackOverflowError e) {
        // not a default-functions expression
      }
    }
    SweepMicro.measure("expr_compile", () -> {
      for (String t : texts) {
        try {
          SweepMicro.sink += JavascriptCompiler.compile(t).variables.length;
        } catch (java.text.ParseException e) {
          throw new IllegalStateException(e);
        }
      }
      return texts.size();
    });
    for (String[] c : EVAL) {
      Expression e = JavascriptCompiler.compile(c[1]);
      DoubleValues[] values = new DoubleValues[e.variables.length];
      for (int i = 0; i < values.length; i++) {
        int k = e.variables[i].charAt(0) - 'a';
        double[] v = new double[DOCS];
        for (int doc = 0; doc < DOCS; doc++) {
          v[doc] = value(doc, k);
        }
        values[i] = new ArrayValues(v);
      }
      SweepMicro.measure(c[0], () -> {
        double sum = 0;
        for (int doc = 0; doc < DOCS; doc++) {
          for (DoubleValues v : values) {
            v.advanceExact(doc);
          }
          sum += e.evaluate(values);
        }
        SweepMicro.sink += (long) sum;
        return DOCS;
      });
    }
  }
}
