import com.carrotsearch.randomizedtesting.RandomizedRunner;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.lucene.tests.analysis.CrankyTokenFilter;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.analysis.MockCharFilter;
import org.apache.lucene.tests.analysis.MockGraphTokenFilter;
import org.apache.lucene.tests.analysis.MockHoleInjectingTokenFilter;
import org.apache.lucene.tests.analysis.MockRandomLookaheadTokenFilter;
import org.apache.lucene.tests.analysis.MockSynonymFilter;
import org.apache.lucene.tests.analysis.MockTokenFilter;
import org.apache.lucene.tests.analysis.MockTokenizer;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BytesRef;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.RunWith;
import org.junit.runner.notification.Failure;

/**
 * Differential fixtures for lucene-test-framework's mock analysis chain (M14.h h3): the tokens
 * Lucene 10.5.0's mocks produce from seeded {@code java.util.Random}s.
 *
 * <p>Writes {@code analysis.txt}: for each of 100 seeds, a {@code seed} line, then for each chain
 * a {@code text <chain> <units>} line, one {@code tok <term> <start> <end> <posInc> <posLen>
 * <type> <payload>} line per token, and an {@code end <finalOffset> <finalPosInc>} line; an
 * {@code exc <message>} line (spaces as {@code _}) records where {@link CrankyTokenFilter} threw,
 * whether from the workflow or from {@code close()}. Each chain draws from its own {@code new
 * Random(seed * 31 + k)} and its text from {@code new Random(seed * 31 + 100 + k)}. Strings
 * travel as their UTF-16 units in hex ({@code -} when empty), payloads as hex ({@code -} when
 * null).
 *
 * <p>{@link MockTokenizer} draws its (output-neutral) read pattern from the randomized-testing
 * context, so the dump runs as a {@link RandomizedRunner} test under {@link JUnitCore}. Run with
 * no {@code tests.*} system properties: {@code rarely} sees multiplier 1, not nightly. Usage: java
 * GenTestFrameworkAnalysis <outdir> [ignored]
 */
public class GenTestFrameworkAnalysis {
  static final StringBuilder OUT = new StringBuilder();
  static final int SEEDS = 100;
  static final String[] SYN_WORDS = {"dogs", "guinea", "pig", "small", "running"};

  static void line(Object... parts) {
    for (int i = 0; i < parts.length; i++) {
      if (i > 0) OUT.append(' ');
      OUT.append(parts[i]);
    }
    OUT.append('\n');
  }

  static String units(CharSequence s) {
    if (s.length() == 0) return "-";
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < s.length(); i++) sb.append(String.format("%04x", (int) s.charAt(i)));
    return sb.toString();
  }

  static String hex(BytesRef b) {
    if (b == null) return "-";
    if (b.length == 0) return "";
    StringBuilder sb = new StringBuilder();
    for (int i = b.offset; i < b.offset + b.length; i++) {
      sb.append(String.format("%02x", b.bytes[i] & 0xFF));
    }
    return sb.toString();
  }

  static void dump(TokenStream ts) throws IOException {
    CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
    OffsetAttribute offset = ts.addAttribute(OffsetAttribute.class);
    PositionIncrementAttribute posInc = ts.addAttribute(PositionIncrementAttribute.class);
    PositionLengthAttribute posLen = ts.addAttribute(PositionLengthAttribute.class);
    TypeAttribute type = ts.addAttribute(TypeAttribute.class);
    PayloadAttribute payload = ts.addAttribute(PayloadAttribute.class);
    try {
      ts.reset();
      while (ts.incrementToken()) {
        line(
            "tok",
            units(term),
            offset.startOffset(),
            offset.endOffset(),
            posInc.getPositionIncrement(),
            posLen.getPositionLength(),
            type.type(),
            hex(payload.getPayload()));
      }
      ts.end();
      line("end", offset.endOffset(), posInc.getPositionIncrement());
    } catch (IOException e) {
      line("exc", e.getMessage().replace(' ', '_'));
    }
    try {
      ts.close();
    } catch (IOException e) {
      line("exc", e.getMessage().replace(' ', '_'));
    }
  }

  static MockTokenizer whitespace(Reader reader) {
    MockTokenizer t = new MockTokenizer(MockTokenizer.WHITESPACE, false);
    t.setReader(reader);
    return t;
  }

  static String text(long seed, int k) {
    return TestUtil.randomAnalysisString(new Random(seed * 31 + 100 + k), 20, false);
  }

  static void analyzer(String chain, Analyzer a, String text) throws IOException {
    line("text", chain, units(text));
    dump(a.tokenStream("f", text));
    a.close();
  }

  static void stream(String chain, String text, TokenStream ts) throws IOException {
    line("text", chain, units(text));
    dump(ts);
  }

  static void seed(long seed) throws IOException {
    line("seed", seed);
    String t;
    analyzer("mock", new MockAnalyzer(new Random(seed * 31)), text(seed, 0));

    Random r1 = new Random(seed * 31 + 1);
    MockAnalyzer simpleStop =
        new MockAnalyzer(r1, MockTokenizer.SIMPLE, true, MockTokenFilter.ENGLISH_STOPSET);
    simpleStop.setMaxTokenLength(TestUtil.nextInt(r1, 1, 10));
    analyzer("simpleStop", simpleStop, text(seed, 1));

    analyzer(
        "keyword",
        new MockAnalyzer(new Random(seed * 31 + 2), MockTokenizer.KEYWORD, false),
        text(seed, 2));

    t = text(seed, 3);
    stream("graph", t, new MockGraphTokenFilter(new Random(seed * 31 + 3), whitespace(new StringReader(t))));

    t = text(seed, 4);
    stream(
        "lookahead",
        t,
        new MockRandomLookaheadTokenFilter(new Random(seed * 31 + 4), whitespace(new StringReader(t))));

    t = text(seed, 5);
    // a throw leaves the tokenizer mid-stream: close() there is the point
    MockTokenizer crankySource = whitespace(new StringReader(t));
    crankySource.setEnableChecks(false);
    stream("cranky", t, new CrankyTokenFilter(crankySource, new Random(seed * 31 + 5)));

    t = text(seed, 6);
    int remainder = new Random(seed * 31 + 6).nextInt(10);
    stream("charFilter", t, whitespace(new MockCharFilter(new StringReader(t), remainder)));

    Random r7 = new Random(seed * 31 + 107);
    StringBuilder syn = new StringBuilder();
    int words = TestUtil.nextInt(r7, 1, 8);
    for (int i = 0; i < words; i++) {
      if (i > 0) syn.append(' ');
      syn.append(SYN_WORDS[r7.nextInt(SYN_WORDS.length)]);
    }
    t = syn.toString();
    stream("synonym", t, new MockSynonymFilter(whitespace(new StringReader(t))));

    t = text(seed, 8);
    stream("holes", t, new MockHoleInjectingTokenFilter(new Random(seed * 31 + 8), whitespace(new StringReader(t))));
  }

  @RunWith(RandomizedRunner.class)
  public static class Dump {
    @Test
    public void dump() throws IOException {
      for (long seed = 0; seed < SEEDS; seed++) seed(seed);
    }
  }

  public static void main(String[] args) throws Exception {
    Result result = JUnitCore.runClasses(Dump.class);
    for (Failure f : result.getFailures()) {
      System.err.println(f.getTrace());
    }
    if (!result.wasSuccessful() || result.getRunCount() != 1) {
      throw new AssertionError("GenTestFrameworkAnalysis: the dump did not run cleanly");
    }
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    Files.writeString(out.resolve("analysis.txt"), OUT.toString(), StandardCharsets.UTF_8);
  }
}
