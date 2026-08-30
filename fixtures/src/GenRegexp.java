import org.apache.lucene.util.automaton.Automaton;
import org.apache.lucene.util.automaton.ByteRunAutomaton;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.automaton.RegExp;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Cross-engine ground truth for {@code lucene-codecs}' {@code regexp.rs}: the
 * accept/reject decision real Lucene's {@link RegExp} grammar makes, pattern by
 * pattern and term by term.
 *
 * <p>Built exactly the way {@code RegexpQuery(Term)} builds it -- {@code
 * RegExp.ALL} syntax flags, no match flags, {@code RegexpQuery.DEFAULT_PROVIDER}
 * (which returns {@code null} for every named automaton), {@link
 * Operations#determinize} at the default work limit, then a {@link
 * ByteRunAutomaton} run over the term's UTF-8 bytes. That last step matters:
 * {@code RegExp} builds a *codepoint* automaton and {@code CompiledAutomaton}
 * converts it to UTF-8 with {@code UTF32ToUTF8}, so this fixture pins the
 * byte-level behaviour a term-dictionary scan actually sees.
 *
 * <p>The pattern list covers every production of the grammar, including the
 * ones a hand-written parser is most likely to get silently wrong: {@code #}
 * (empty language) and {@code @} (any string) as *operators* rather than
 * literals, {@code &} intersection, {@code "..."} quoted literals, {@code
 * <n-m>} numeric intervals in both their fixed-width and any-width forms, the
 * {@code \d \D \s \S \w \W} predefined classes inside and outside {@code
 * [...]}, {@code ~} as an ordinary character (complement is *not* in {@code
 * RegExp.ALL}), a leading quantifier as an ordinary character, stacked
 * quantifiers ({@code a**}), and multi-byte codepoints under {@code .} and
 * {@code [^...]}.
 *
 * <p>Two files are written, both plain text so the Rust side needs no parser:
 * {@code terms.txt} is one term per line (the first line is deliberately
 * empty -- the empty term is a real case), and {@code cases.tsv} is one line
 * per pattern, {@code <pattern> TAB <accept-mask>}, where the mask is one
 * {@code '0'} or {@code '1'} per line of {@code terms.txt}. A pattern real
 * Lucene rejects gets the literal mask {@code ERR} instead. No pattern or term
 * here contains a tab or a newline.
 */
public class GenRegexp {

  static final String[] TERMS = {
    "", "a", "aa", "ab", "abc", "cat", "cats", "ca", "dog", "bird", "*cat", "a~b",
    "a&b", "a*b", "a.b", "5", "05", "40", "41", "007", "10", "1", "123", "x2y",
    "wifi", "AB", "a1", "1a", "abab", "ababab", "aaa", "aaaa", "possible",
    "café", "€", "a€c", "|a", "&a", "a b", "_", "-", "\\",
    "cat7", "catalog", "izat", "argument", "technolog", "0000000009",
    "999999999999999999999", "]", "z", "-z", "abd", "abdacd", "cababt", "caat",
    "possibl", "cd", "adb", "at", "a*", "a?", "a+", "{2,3}", "+x", "?"
  };

  static final String[] PATTERNS = {
    // Literals, `.`, and the whole-term-match convention.
    "cat", "ca", "c.t", ".", "a.c", ".*",
    // Quantifiers, including stacked ones and a leading one (which is an
    // ordinary character in this grammar, not a parse error).
    "ca*t", "ca+t", "ca?t", "a**", "a?+", "a{2}?", "*cat", "{2,3}", "+x", "?",
    "a{3}", "a{0,0}b", "a{2,}", "a{2,4}", "a{0,2}b", "a{2,3}b*", "(ab){2,3}",
    "(a?)*", "(a?)*b", "ca{2,3}t", "c(ab){2,}t",
    // Character classes, ranges, negation and escapes inside them.
    "[cb]at", "[a-c]at", "[^ab]at", "[^a]", "[\\-z]", "[\\]]",
    // Predefined classes, inside and outside a class.
    "\\d+", "\\D", "\\w+", "\\W", "\\s", "\\S", "[\\dx]+", "[^\\d]", "\\\\",
    "[\\\\]",
    // Union, intersection and their precedence against concatenation.
    "cat|dog", "a|b", "[a-z]+&...", "(cat|dog)&(dog|bird)", "ab&ab|cd", "a.&.b",
    "cat.*&catalog",
    // Grouping, the empty group, and quoted literal strings.
    "(cat)+", "(cat|dog)s", "(a(b|c)d)+", "()", "a()b", "\"a*b\"", "\"\"",
    "\"cat\"[0-9]",
    // The optional-syntax operators `RegExp.ALL` turns on.
    "#", "#|cat", "a#b", "@", "cat@",
    "<05-40>", "<5-40>", "<40-5>", "x<1-3>y", "<1-30>",
    // `~` is a literal: DEPRECATED_COMPLEMENT is not part of RegExp.ALL.
    "a~b", "~",
    // Escaped operators, and the operators in leading position.
    "a\\*b", "a\\.b", "a\\[b", "a\\&b", "a\\@b", "a\\#b", "|a", "&a",
    // The empty pattern matches only the empty term.
    "",
    // Malformed patterns real Lucene rejects.
    "(cat", "cat)", "[cat", "[]", "[^]", "[a-]", "a{2,3", "a{x}", "a{}",
    "a{3,2}", "ab\\", "[a\\", "\\A", "\\q", "[\\A]", "\"abc", "<-3>", "<1->",
    "<1-2-3>", "<a-b>", "<1-2", "<name>", "a|", "a&"
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("regexp");
    Files.createDirectories(out);

    Files.writeString(out.resolve("terms.txt"), String.join("\n", TERMS) + "\n");

    StringBuilder cases = new StringBuilder();
    for (String pattern : PATTERNS) {
      cases.append(pattern).append('\t').append(mask(pattern)).append('\n');
    }
    Files.writeString(out.resolve("cases.tsv"), cases.toString());

    System.out.println("wrote regexp/ fixture directory");
  }

  private static String mask(String pattern) {
    ByteRunAutomaton run;
    try {
      // Exactly `RegexpQuery(Term)`: RegExp.ALL, no match flags, a provider
      // that knows no named automata, determinized at the default work limit.
      RegExp regexp = new RegExp(pattern, RegExp.ALL);
      Automaton automaton =
          Operations.determinize(
              regexp.toAutomaton(name -> null), Operations.DEFAULT_DETERMINIZE_WORK_LIMIT);
      run = new ByteRunAutomaton(automaton);
    } catch (RuntimeException e) {
      return "ERR";
    }
    StringBuilder mask = new StringBuilder(TERMS.length);
    for (String term : TERMS) {
      byte[] bytes = term.getBytes(StandardCharsets.UTF_8);
      mask.append(run.run(bytes, 0, bytes.length) ? '1' : '0');
    }
    return mask.toString();
  }
}
