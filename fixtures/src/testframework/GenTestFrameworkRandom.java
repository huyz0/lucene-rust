import com.carrotsearch.randomizedtesting.generators.RandomNumbers;
import com.carrotsearch.randomizedtesting.generators.RandomPicks;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.tests.util.English;
import org.apache.lucene.tests.util.LineFileDocs;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.tests.util.automaton.AutomatonTestUtil;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.automaton.Automaton;

/**
 * Differential fixtures for lucene-test-framework's random generators (M14.h): what Lucene
 * 10.5.0's test framework draws from a seeded {@code java.util.Random}.
 *
 * <p>Writes {@code random.txt}: for each of 200 seeds, a {@code seed} line, then one line per
 * value, {@code <category> <name> <value>}. Each category draws from its own {@code new
 * Random(seed * 31 + k)}, so one divergence does not cascade into the next category. Strings
 * travel as their UTF-16 units in hex ({@code -} when empty), byte arrays as hex, code point
 * lists comma-joined.
 *
 * <p>{@code LineFileDocs} reads the gzip excerpt the crate ships ({@code
 * crates/lucene-test-framework/data/}) and, for the plain-file path, its gunzipped copy, which
 * this program writes to {@code <outdir>/../europarl.lines.excerpt.txt} on the fly and deletes.
 * The Rust test gunzips the same excerpt itself.
 *
 * <p>Run with no {@code tests.*} system properties: {@code atLeast}/{@code rarely} see
 * multiplier 1, not nightly. Usage: java GenTestFrameworkRandom <outdir> <excerpt.txt.gz>
 */
public class GenTestFrameworkRandom {
  static final StringBuilder OUT = new StringBuilder();
  static final int SEEDS = 200;
  static final int LINE_DOC_SEEDS = 40;
  static final int LINE_DOCS = 60;

  static void line(Object... parts) {
    for (int i = 0; i < parts.length; i++) {
      if (i > 0) OUT.append(' ');
      OUT.append(parts[i]);
    }
    OUT.append('\n');
  }

  static String units(String s) {
    if (s.isEmpty()) return "-";
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < s.length(); i++) sb.append(String.format("%04x", (int) s.charAt(i)));
    return sb.toString();
  }

  static String units(char[] c) {
    return units(new String(c));
  }

  static String hex(byte[] b, int off, int len) {
    if (len == 0) return "-";
    StringBuilder sb = new StringBuilder();
    for (int i = off; i < off + len; i++) sb.append(String.format("%02x", b[i] & 0xFF));
    return sb.toString();
  }

  static String ints(int[] a) {
    if (a.length == 0) return "-";
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < a.length; i++) {
      if (i > 0) sb.append(',');
      sb.append(a[i]);
    }
    return sb.toString();
  }

  static void numbers(Random r) {
    line("num", "nextInt", TestUtil.nextInt(r, -100, 100));
    line("num", "nextIntWide", TestUtil.nextInt(r, 0, Integer.MAX_VALUE));
    line("num", "nextLong", TestUtil.nextLong(r, -1000, 1000));
    line("num", "nextLongFull", TestUtil.nextLong(r, Long.MIN_VALUE, Long.MAX_VALUE));
    line("num", "nextLongBig", TestUtil.nextLong(r, -5, Long.MAX_VALUE / 3));
    line("num", "randomIntBetween", RandomNumbers.randomIntBetween(r, -5, 5));
    line("num", "randomIntBetweenFull",
        RandomNumbers.randomIntBetween(r, Integer.MIN_VALUE, Integer.MAX_VALUE));
    line("num", "randomLongBetween", RandomNumbers.randomLongBetween(r, -10, 1L << 40));
    line("num", "randomLongBetweenFull",
        RandomNumbers.randomLongBetween(r, Long.MIN_VALUE, Long.MAX_VALUE));
    line("num", "randomLongBetweenHigh",
        RandomNumbers.randomLongBetween(r, Long.MAX_VALUE - 3, Long.MAX_VALUE));
    byte[] big = TestUtil.nextBigInteger(r, 20).toByteArray();
    line("num", "nextBigInteger", hex(big, 0, big.length));
    line("num", "randomFrom",
        RandomPicks.randomFrom(r, List.of("a", "b", "c", "d", "e", "f", "g")));
    line("num", "atLeast", LuceneTestCase.atLeast(r, 10));
    for (int i = 0; i < 20; i++) line("num", "rarely", LuceneTestCase.rarely(r));
    line("num", "usually", LuceneTestCase.usually(r));
  }

  static void strings(Random r) {
    line("str", "simple", units(TestUtil.randomSimpleString(r)));
    line("str", "simpleMax", units(TestUtil.randomSimpleString(r, 7)));
    line("str", "simpleBetween", units(TestUtil.randomSimpleString(r, 3, 12)));
    line("str", "simpleRange", units(TestUtil.randomSimpleStringRange(r, 'k', 'q', 9)));
    line("str", "unicode", units(TestUtil.randomUnicodeString(r)));
    line("str", "unicodeMax", units(TestUtil.randomUnicodeString(r, 50)));
    char[] fixed = new char[20];
    TestUtil.randomFixedLengthUnicodeString(r, fixed, 0, fixed.length);
    line("str", "fixedLength", units(fixed));
    line("str", "regexpish", units(TestUtil.randomRegexpishString(r)));
    line("str", "regexpishMax", units(TestUtil.randomRegexpishString(r, 40)));
    line("str", "htmlish", units(TestUtil.randomHtmlishString(r, 8)));
    line("str", "realistic", units(TestUtil.randomRealisticUnicodeString(r)));
    line("str", "realisticMax", units(TestUtil.randomRealisticUnicodeString(r, 30)));
    line("str", "realisticBetween", units(TestUtil.randomRealisticUnicodeString(r, 5, 30)));
    line("str", "fixedByteLength", units(TestUtil.randomFixedByteLengthUnicodeString(r, 25)));
    BytesRef term = TestUtil.randomBinaryTerm(r);
    line("str", "binaryTerm", hex(term.bytes, term.offset, term.length));
    BytesRef term7 = TestUtil.randomBinaryTerm(r, 7);
    line("str", "binaryTermLen", hex(term7.bytes, term7.offset, term7.length));
    line("str", "analysisSimple", units(TestUtil.randomAnalysisString(r, 30, true)));
    line("str", "analysis", units(TestUtil.randomAnalysisString(r, 30, false)));
    line("str", "subString", units(TestUtil.randomSubString(r, 10, false)));
    line("str", "subStringSimple", units(TestUtil.randomSubString(r, 10, true)));
    // ASCII and Latin-1 only: the case tables of JDK 21 and the Rust port's
    // newer Unicode agree on these.
    line("str", "recase", units(TestUtil.randomlyRecaseCodePoints(r, "Hello Wörld, ÀÉÎ ßtraße 42")));
  }

  static void english(Random r) {
    line("eng", "int", units(English.intToEnglish(r.nextInt())));
    line("eng", "small", units(English.intToEnglish(r.nextInt(2000) - 1000)));
    line("eng", "long", units(English.longToEnglish(r.nextLong())));
  }

  static void automata(Random r) {
    line("aut", "regexp", units(AutomatonTestUtil.randomRegexp(r)));
    Automaton a = AutomatonTestUtil.randomAutomaton(r);
    int transitions = 0;
    for (int s = 0; s < a.getNumStates(); s++) transitions += a.getNumTransitions(s);
    line("aut", "automaton", a.getNumStates(), transitions, a.isDeterministic());
    AutomatonTestUtil.RandomAcceptedStrings ras;
    try {
      ras = new AutomatonTestUtil.RandomAcceptedStrings(a);
    } catch (IllegalArgumentException e) {
      line("aut", "accepted", "ERR");
      return;
    }
    for (int i = 0; i < 3; i++) {
      try {
        line("aut", "accepted", ints(ras.getRandomAcceptedString(r)));
      } catch (RuntimeException e) {
        // Dead states (RuntimeException) or a surrogate-only transition
        // (IllegalArgumentException).
        line("aut", "accepted", "ERR");
        return;
      }
    }
  }

  static void lineDocs(String kind, Random r, Path file) throws Exception {
    try (LineFileDocs docs = new LineFileDocs(r, file.toString())) {
      for (int i = 0; i < LINE_DOCS; i++) {
        Document d = docs.nextDoc();
        line("docs", kind, d.get("docid"), d.getField("docid_int").numericValue(),
            d.getField("page_views").numericValue(), units(d.get("title")), d.get("date"),
            d.get("body").hashCode());
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Path gz = Path.of(args[1]);
    Files.createDirectories(out);
    Path plain = out.resolveSibling("europarl.lines.excerpt.txt");
    try (var in = new java.util.zip.GZIPInputStream(Files.newInputStream(gz))) {
      Files.write(plain, in.readAllBytes());
    }
    try {
      for (int seed = 0; seed < SEEDS; seed++) {
        line("seed", seed);
        long base = seed * 31L;
        numbers(new Random(base));
        strings(new Random(base + 1));
        english(new Random(base + 2));
        automata(new Random(base + 3));
        if (seed < LINE_DOC_SEEDS) {
          lineDocs("gz", new Random(base + 4), gz);
          lineDocs("plain", new Random(base + 5), plain);
        }
      }
    } finally {
      Files.delete(plain);
    }
    Files.writeString(out.resolve("random.txt"), OUT.toString());
  }
}
