import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * M12 T12.7: {@code java.util.regex} beyond the shim -- the backtracking engine of {@code
 * crates/lucene-analysis/src/util/java_backtrack.rs} against {@code Pattern} itself ({@code
 * tests/java_regex_fixtures.rs} compares).
 *
 * <ul>
 *   <li>{@code curated.txt}: {@link #CURATED} (every construct the shim refuses, and its corner
 *       cases) over {@link #INPUTS};
 *   <li>{@code random.txt}: 2,500 patterns drawn from a grammar of those constructs (groups,
 *       alternation, lookaround, atomic groups, backreferences, greedy, lazy and possessive
 *       quantifiers, anchors, boundaries, inline flags, classes) over 8 random inputs each;
 *   <li>{@code deep.txt}: inputs of 300 to 1,500 characters, deep enough that the matcher's
 *       backtracking outgrows the caller's stack budget and is retried (an overflow inside a
 *       negated lookaround or a zero-count quantifier must not read as a match): {@link #DEEP}
 *       over {@code "ab"} repeated then {@code "c"}, and 200 generated patterns over one long
 *       input each. An input on which Java itself overflows is written as {@code SOE}, and
 *       skipped.
 * </ul>
 *
 * Each block is {@code P<TAB>pattern} then {@code I<TAB>input<TAB>result}, the result {@link
 * GenAnalysisCommon#regexRun}'s ({@code find()} spans with every group, {@code replaceAll("<$0>")},
 * {@code matches()}) or the exception. Inputs keep to characters whose properties JDK 21 and 25
 * agree on.
 */
public class GenJavaRegex {

  static final String[] CURATED = {
    // backreferences
    "(a)\\1", "(a*)\\1", "(\\w)\\1+", "(?<x>ab)\\k<x>", "(a)|\\1", "(a)?\\1", "(?i)(a)\\1", "(?iu)(é)\\1",
    "\\1(a)", "(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)\\10", "(a)\\11", "(a)(b)\\2\\1", "((a)b)\\2",
    // lookaround
    "(?=a)", "(?=a)a", "a(?=b)", "a(?!b)", "(?!a).", "(?<=a)b", "(?<!a)b", "(?<=ab|c)d", "(?<=a{1,3})b",
    "(?<=\\w)\\b", "(?=(a))", "(?=(a+))a", "(?!(a))b", "x(?=a|ab)", "(?<=^|,)\\w+", "(?<=😀)a", "(?=😀).",
    // possessive and atomic
    "a++", "a*+a", "a?+a", "a{1,2}+a", "(ab)++", "(?>a+)a", "(?>a|ab)c", "(?>(a))\\1", "[ab]*+b", ".*+",
    // boundaries
    "\\b", "\\B", "\\bfox", "\\b\\w+\\b", "\\Ba", "a\\b", "\\b.", "é\\b", "\\b\u0301", "(?U)\\b\\w+",
    // anchors
    "(?m)^a", "(?m)a$", "(?m)^", "(?m)$", "(?m)^$", "(?m)$\\n", "\\Z", "a\\Z", "\\z", "\\A", "\\G", "\\Ga",
    "\\G\\w", "(?d)$", "(?d).", "(?md)^\\w", "(?m)(?d)a$", "$\\r", "\\r$", "(?m)\\r$",
    // line breaks and graphemes
    "\\R", "\\R+", "a\\Rb", "\\X", "\\X+",
    // flags
    "(?x) a b", "(?x)a\\ b", "(?x)a#c\nb", "(?x)[a b]", "(?U)\\w", "(?U)\\d+", "(?U)\\s", "(?U)\\p{Lower}",
    "(?U)[[:alpha:]]", "(?iU)k", "(?s).", "(?-s).",
    // properties
    "\\p{IsLatin}", "\\p{IsGreek}+", "\\p{InGreek}", "\\p{InBasicLatin}+", "\\p{IsAlphabetic}",
    "\\p{IsLetter}", "\\p{IsIdeographic}", "\\p{IsLowercase}", "\\p{IsUppercase}", "\\p{IsWhite_Space}",
    "\\p{IsDigit}", "\\p{IsPunctuation}", "\\p{javaLowerCase}", "\\p{javaUpperCase}", "\\p{javaWhitespace}",
    "\\p{javaLetterOrDigit}", "\\p{script=Latin}", "\\p{sc=Grek}", "\\p{block=Greek}", "\\p{blk=Latin_1_Supplement}",
    "\\p{IsHan}", "\\p{IsCommon}", "\\P{IsLatin}+", "[\\p{IsLatin}&&\\p{Lu}]",
    // escapes
    "\\cA", "\\cZ", "\\0101", "\\0377", "\\08", "\\x41", "\\N{LATIN SMALL LETTER A}",
    // repetition corners
    "(a|)*", "(a*)+", "(a*)*", "(a|b)*?", "(|a)+", "(a?)+?", "(a|)+b", "((a)|b)*", "(a*)+b", "(?:a|)*",
    "(a{0,2})*", "(a|ab)(c|bcd)(d*)", "(a+|b)*", "(a+|b){0,}", "(a+|b)+", "(a+|b)?", "([ab]*?)(b)?",
    "(.*)c(.*)", "(?:(a)|b)*", "(a)|b",
    // a greedy unbounded group loop no quantified group encloses remembers, for the whole find,
    // where another iteration failed (so a failed start leaves different captures behind)
    "\\b|(?:(.b?)*(b)*+\\G)", "\\b|(?:(.b?)*(b)*+\\G\\1)", "\\b|(?:(?:(.b?)*)?(b)*+\\G)",
    "\\b|(?:(?:(.b?)*){1}(b)*+\\G)", "\\b|(?=(.b?)*(b)*+\\G)", "\\b|(?>(.b?)*(b)*+\\G)",
    "\\b|(?:(.b?){1,}(b)*+\\G)", "\\b|(?:(.b?){2,}(b)*+\\G)", "\\b|(?:(.b?){0,100}(b)*+\\G)",
    "\\b|(?:(?<n>.b?)*(b)*+\\G\\k<n>)", "\\b|(?:(?>(.b?)*)*(b)*+\\G)", "\\b|(?:(?i:(.b?)*)*(b)*+\\G)",
    "\\b|(?:(.b?)*?(b)*+\\G)", "\\b|(?:((.b?)*)*+(b)*+\\G)", "\\b|(?:(.b?)*(.b?)*(b)*+\\G)",
    // an escaped supplementary character selects code-point stepping only on its own
    "\\x{1F601}|\\B", "\\x{1F601}a|\\B", "(?i)\\x{1F601}|\\B", "\\x{1F601}+|\\B", "\\Q😁\\E|\\B",
    // comments mode reaches past whitespace for the `?` of a group
    "(?x)( ?)", "(?x)( ?:a)+", "(?x)( ?<n> a)", "(?x)(? :a)", "(?x)( {2})", "(?x)( *)",
    // rejected by Java
    "(", "(?<=a*)b", "(?<=a+)b", "\\k<x>", "(a)\\k<y>", "*", "a**", "a{2,1}", "\\p{Latin}", "(?<a_b>x)",
  };

  static final String[] INPUTS = {
    "", "a", "aa", "aaa", "ab", "abab", "ba", "abc", "aab", "abcd", "a\nb", "a\n", "\na", "a\r\nb\r\n",
    "x\u2028y", "fox foxes", "a_1 b", "é a", "e\u0301 x", "😀a", "a😀b", "AbA", "αβγ ΑΒ", "中文abc",
    "a,b,c", "a\u0085", "  a\tb", "a b\u000Bc", "xbb0bbub0bba", "Bbb0\uFFFDbunBa\uFFFD\uFFFD",
  };

  /** Patterns whose deep backtracking sits under a negation or a zero-count quantifier. */
  static final String[] DEEP = {
    "(?!(?:a|b)*c)", "(?>(?:a|b)*c)?", "((?:a|b)*c)?+", "(?:a|b)*c", "(?=(?:a|b)*c)", "(?!(a|b)*+c)",
    "(?<!x)(?:(?:a|b)*c)?+x?", "((?:a|b)*c){0,1}", "(?:(?:a|b)*c)*", "(a|b)*c|(?!(?:a|b)*c)",
  };

  static String esc(String s) {
    return AnalysisRows.esc(s);
  }

  static final String[] ATOMS = {
    "a", "b", "c", "ab", ".", "\\w", "\\d", "\\s", "[ab]", "[^a]", "[a-c&&[^b]]", "\\b", "\\B", "^", "$",
    "é", "😀", "\\p{L}", "\\p{IsLatin}", "\\R", "\\Z", "\\z", "\\G", " ", "a{0,3}", "\\S", "\\W", "\\h", "\\v", "\\p{Lu}", "\\x{1F600}", "\\Q.\\E", "\\.",
    "[\\w&&[^a]]", "(?<n>a)", "\\uFFFD"
  };
  static final String[] QUANTS = {"*", "+", "?", "{2}", "{1,2}", "{0,}", "{2,}", "{0,1}", "{0,2}", "*", "*"};
  static final String[] FLAGS = {"(?i)", "(?m)", "(?s)", "(?x)", "(?iu)", "(?d)", "(?U)"};

  static String gen(Random r, int depth, int[] groups) {
    StringBuilder b = new StringBuilder();
    int n = 1 + r.nextInt(3);
    for (int k = 0; k < n; k++) {
      int c = r.nextInt(depth > 2 ? 6 : 12);
      String atom;
      if (c < 6) {
        atom = ATOMS[r.nextInt(ATOMS.length)];
      } else if (c == 6) {
        groups[0]++;
        atom = "(" + gen(r, depth + 1, groups) + ")";
      } else if (c == 7) {
        atom = "(?:" + gen(r, depth + 1, groups) + "|" + gen(r, depth + 1, groups) + ")";
      } else if (c == 8) {
        String[] la = {"(?=", "(?!", "(?<=", "(?<!", "(?>"};
        String kind = la[r.nextInt(la.length)];
        // lookbehind bodies stay bounded: literals and classes only
        String body = kind.startsWith("(?<") ? ATOMS[r.nextInt(12)] + (r.nextBoolean() ? "" : ATOMS[r.nextInt(12)]) : gen(r, depth + 1, groups);
        atom = kind + body + ")";
      } else if (c == 9 && groups[0] > 0) {
        atom = "\\" + (1 + r.nextInt(groups[0]));
      } else if (c == 10) {
        atom = FLAGS[r.nextInt(FLAGS.length)];
      } else {
        groups[0]++;
        atom = "(" + gen(r, depth + 1, groups) + "|" + gen(r, depth + 1, groups) + ")";
      }
      b.append(atom);
      if (r.nextInt(3) == 0 && !atom.startsWith("(?") || r.nextInt(8) == 0) {
        b.append(QUANTS[r.nextInt(QUANTS.length)]);
        int m = r.nextInt(4);
        if (m == 1) b.append('?');
        if (m == 2) b.append('+');
      }
    }
    return b.toString();
  }

  static final String ALPHABET = "aabbc_1 \n\r,é😀ΑAB.\uFFFD\u0301\u2028Ω";

  static String input(Random r) {
    StringBuilder b = new StringBuilder();
    int n = r.nextInt(9);
    for (int i = 0; i < n; i++) b.appendCodePoint(ALPHABET.codePointAt(ALPHABET.offsetByCodePoints(0, r.nextInt(ALPHABET.codePointCount(0, ALPHABET.length())))));
    return b.toString();
  }

  static void block(StringBuilder out, String p, List<String> inputs) {
    out.append("P\t").append(esc(p)).append('\n');
    for (String in : inputs) {
      String result;
      try {
        result = GenAnalysisCommon.regexRun(p, in);
      } catch (StackOverflowError e) {
        result = "SOE";
      }
      out.append("I\t").append(esc(in)).append('\t').append(result).append('\n');
    }
  }

  /** {@code base} repeated to {@code length} characters. */
  static String repeatTo(String base, int length) {
    StringBuilder b = new StringBuilder();
    while (b.length() < length) b.append(base.isEmpty() ? "a" : base);
    String s = b.substring(0, length);
    // Never a lone high surrogate at the cut.
    return Character.isHighSurrogate(s.charAt(length - 1)) ? s.substring(0, length - 1) : s;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("java_regex");
    Files.createDirectories(out);
    StringBuilder c = new StringBuilder();
    for (String p : CURATED) block(c, p, List.of(INPUTS));
    Files.writeString(out.resolve("curated.txt"), c.toString(), StandardCharsets.UTF_8);
    Random r = new Random(2027);
    StringBuilder rnd = new StringBuilder();
    for (int i = 0; i < 2500; i++) {
      String p = gen(r, 0, new int[1]);
      List<String> ins = new ArrayList<>();
      for (int k = 0; k < 8; k++) ins.add(input(r));
      block(rnd, p, ins);
    }
    Files.writeString(out.resolve("random.txt"), rnd.toString(), StandardCharsets.UTF_8);
    StringBuilder deep = new StringBuilder();
    for (String p : DEEP) block(deep, p, List.of("ab".repeat(400) + "c", "ab".repeat(700) + "c"));
    Random d = new Random(2028);
    for (int i = 0; i < 200; i++) {
      String p = gen(d, 0, new int[1]);
      block(deep, p, List.of(repeatTo(input(d) + input(d), 300 + d.nextInt(1201))));
    }
    Files.writeString(out.resolve("deep.txt"), deep.toString(), StandardCharsets.UTF_8);
  }
}
