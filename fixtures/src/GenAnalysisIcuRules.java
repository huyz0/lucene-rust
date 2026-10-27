import com.ibm.icu.text.BreakIterator;
import com.ibm.icu.text.RuleBasedBreakIterator;
import java.nio.charset.StandardCharsets;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.TreeMap;

/**
 * (Rule sets of a custom {@code $dictionary} variable are not generated: ICU4J 77.1 loops forever
 * iterating one over Thai text; Lucene's {@code Default.rbbi} covers the dictionary categories.)
 *
 * M12 T12.4: ICU4J 77.1's break rule compiler ({@code new RuleBasedBreakIterator(String rules)},
 * what Lucene's {@code ICUTokenizerFactory} runs for each of its {@code rulefiles}), compared by
 * {@code crates/lucene-analysis-icu/tests/icu_rule_compiler_fixtures.rs} with this port's
 * compiler.
 *
 * <p>{@code analysis_icu_rules/*.rbbi}: the rule sources -- the five {@code .rbbi} files of
 * Lucene 10.5.0's analysis-icu (Default, MyanmarSyllable, and the three of its tokenizer factory
 * test), copied from {@code fixtures/corpus/analysis-icu-rules/}, and {@code inline_*.rbbi}, written here:
 * each ICU rule feature, and rule sets ICU rejects. {@code texts.txt}: the corpus, UTF-16 units in
 * hex. {@code <name>.tsv}: per text, the boundaries {@code first()}..{@code next()} as {@code
 * position:ruleStatus:statusVector}, or a single line {@code ERROR <message>} when ICU4J refuses
 * the rules.
 */
public class GenAnalysisIcuRules {
  static final String BS = "\\";

  static final int[] POOL = {
    'a', 'B', 'z', ' ', ' ', '.', ',', '\'', '"', '(', ')', '-', '1', '9', '_', '\n', '\r', 0x2029,
    0x3042, 0x30a2, 0x4e00, 0x4e2d, 0xac00, 0xe01, 0xe32, 0xe40, 0xe48, 0x627, 0x5d0, 0x301, 0x200d,
    0x1f600, 0x1f1fa, 0x1f1f8, 0xa0, 0x3b1, 0x915, 0x94d, 0x1000, 0x103a, 0x10400
  };

  static String hex(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      if (i > 0) b.append(' ');
      b.append(String.format("%04x", (int) s.charAt(i)));
    }
    return b.toString();
  }

  /** The inline rule sets, by name: each ICU rule feature, then the ones ICU rejects. */
  static TreeMap<String, String> inline() {
    TreeMap<String, String> m = new TreeMap<>();
    m.put("any", ".;");
    m.put("plus_tag", ".+ {200};");
    m.put("var_sets", "$L = [a-z]; $D = [0-9]; $L+ {100}; $D+ {200}; . ;");
    m.put(
        "var_chain",
        "$A = [a-c]; $B = $A; $C = $B; $C+ {1}; $A | [x-z] {2}; . ;");
    m.put(
        "var_expr",
        "$Word = [a-z]+; $Num = [0-9]+; $Sep = [ ,.]; ($Word | $Num) ($Sep ($Word | $Num))* {5}; . ;");
    m.put("operators", "[a-c] [d-f]? [g-i]* [j-l]+ {3}; ([m-o] [p-r])+ {4}; (a|b) c {6}; . ;");
    m.put("lookahead", "[a-z]+ / '.' {10}; [a-z]+ {20}; . ;");
    m.put("lookahead_two", "a b / c d {1}; a b {2}; [a-z] {3}; . ;");
    m.put("bof", "^[a-z]+ {7}; [a-z]+ {8}; . ;");
    m.put("bof_two", "^a {1}; ^b+ {2}; [a-z]+ {3}; . ;");
    m.put("chain", "!!chain; [a-z]+ '-' ; [a-z]+ {1}; . ;");
    m.put("chain_nochain", "!!chain; $L = [a-z]; ^$L+ '-' ; $L+ {1}; . ;");
    m.put("forward_option", "!!forward; [a-z]+ {1}; . ;");
    m.put("hard_break", "!!lookAheadHardBreak; [a-z]+ / '.' {10}; [a-z]+ {20}; . ;");
    m.put("tags", "[a-c]+ {100}; [a-c]+ [d-f]+ {200}; [d-f]+ {100}{300}; . ;");
    m.put("tags_two_in_state", "[a-b]+ {1}; [a-b]+ [c] {2}; [a-c] {3}; . ;");
    m.put("quote_text", "'ab' 'c' {1}; 'a''b' {2}; [a-z] ; . ;");
    m.put("quote_set", "'[a-c]' {1}; [a-c] {2}; . ;");
    m.put("escapes", BS + "x41 {1}; " + BS + "x{42} {2}; " + BS + "p{L}+ {3}; [" + BS + "p{N}]+ {4}; . ;");
    m.put("comments", "# a comment\n[a-z]+ {1}; # trailing\n\n. ; # last\n");
    m.put("quoted_literals_only", "!!quoted_literals_only; 'a' 'b' {1}; [c] {2}; . ;");
    m.put("unquoted_literals", "!!unquoted_literals; a b {1}; [c] {2}; . ;");
    m.put("negated_sets", "[^a-c] {1}; [a-c]+ {2}; . ;");
    m.put("set_ops", "[[a-z]-[aeiou]]+ {1}; [[a-z]&[aeiou]]+ {2}; [0-9[a-f]]+ {3}; . ;");
    m.put("props", BS + "p{Lu}" + BS + "p{Ll}+ {1}; [:Nd:]+ {2}; " + BS + "p{Script=Greek}+ {3}; . ;");
    m.put("eof_set", "$X = [a-z]; $X+ {1}; . ;");
    m.put("empty_rule_set", "$x = [a-z]; ");
    m.put("same_set_twice", "[a-z]+ {1}; [a-z]+ [0-9] {2}; [a-z] {3}; . ;");
    m.put("nested_parens", "(((a|b)c)|d)+ {1}; . ;");
    m.put("star_of_group", "(ab)* {1}; . ;");
    m.put("question_cat", "a? b? c? {1}; . ;");
    // rules ICU rejects: the code and the position are the test
    m.put("err_unclosed_set", "[a-z ;");
    m.put("err_undefined_var", "$nope+ ; . ;");
    m.put("err_var_redefined", "$a = [a]; $a = [b]; . ;");
    m.put("err_assign_expr", "$a = [a] ;; $a = ;");
    m.put("err_paren", "(a b ; . ;");
    m.put("err_rparen", "a b) ; . ;");
    m.put("err_option", "!!bogus; . ;");
    m.put("err_tag", "a {x}; . ;");
    m.put("err_tag_open", "a {12 ; . ;");
    m.put("err_tag_big", "a {99999999999}; . ;");
    m.put("err_no_semicolon", "a b");
    m.put("err_empty_set", "[] ; . ;");
    m.put("err_empty_strings_set", "[{ab}] ; . ;");
    m.put("err_syntax", "= a ;");
    m.put("err_quote_newline", "'abc\ndef' ;");
    m.put("err_bad_escape", BS + "x{zz} ;");
    m.put("err_var_name", "$ = [a]; . ;");
    m.put("err_var_not_set", "$a = b c; [$a] ; . ;");
    m.put("err_empty", "");
    m.put("err_only_comment", "# nothing\n");
    return m;
  }

  static void run(Path out, String name, String rules, List<String> texts) throws Exception {
    StringBuilder sb = new StringBuilder();
    RuleBasedBreakIterator bi;
    try {
      bi = new RuleBasedBreakIterator(rules);
    } catch (Throwable t) {
      sb.append("ERROR ").append(t.getClass().getSimpleName()).append(": ").append(t.getMessage()).append('\n');
      Files.writeString(out.resolve(name + ".tsv"), sb.toString(), StandardCharsets.UTF_8);
      return;
    }
    for (String s : texts) {
      bi.setText(s);
      StringBuilder row = new StringBuilder();
      for (int b = bi.first(); b != BreakIterator.DONE; b = bi.next()) {
        if (row.length() > 0) row.append(' ');
        int[] v = new int[bi.getRuleStatusVec(null)];
        bi.getRuleStatusVec(v);
        row.append(b).append(':').append(bi.getRuleStatus()).append(':');
        for (int k = 0; k < v.length; k++) row.append(k > 0 ? "," : "").append(v[k]);
      }
      sb.append(row).append('\n');
    }
    Files.writeString(out.resolve(name + ".tsv"), sb.toString(), StandardCharsets.UTF_8);
  }

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args[0]);
    Path out = data.resolve("analysis_icu_rules");
    Files.createDirectories(out);
    List<String> texts = new ArrayList<>(AnalysisRows.corpus("analysis-icu.txt"));
    Collections.addAll(
        texts, "", "a", " ", "abc def ghi", "abc.", "abc. def", "ab cd", "abcd", "abc-def-ghi",
        "aaa bbb ccc", "xay", "1a2b3c", "a1 b2", "x.y.z", "Hello world, 3.14 can't stop.",
        "ab'cd", "a\"b", "A1b2", "ABc", "abcdefghijklmnopqrstuvwxyz", "0123456789",
        "การ ab เก", "xyการzz");
    Random rnd = new Random(0x72756c65L);
    for (int i = 0; i < 160; i++) {
      StringBuilder sb = new StringBuilder();
      int len = 1 + rnd.nextInt(20);
      for (int j = 0; j < len; j++) sb.appendCodePoint(POOL[rnd.nextInt(POOL.length)]);
      texts.add(sb.toString());
    }
    // the small alphabets the inline rules are written over
    String alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 .,-'";
    for (int i = 0; i < 160; i++) {
      StringBuilder sb = new StringBuilder();
      int len = 1 + rnd.nextInt(16);
      for (int j = 0; j < len; j++) sb.append(alphabet.charAt(rnd.nextInt(alphabet.length())));
      texts.add(sb.toString());
    }
    StringBuilder t = new StringBuilder();
    for (String s : texts) t.append(hex(s)).append('\n');
    Files.writeString(out.resolve("texts.txt"), t.toString(), StandardCharsets.UTF_8);

    // Lucene's own rule files, compiled as ICUTokenizerFactory.parseRules reads them
    Path src =
        Path.of(System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus"), "analysis-icu-rules");
    List<Path> files = new ArrayList<>();
    try (var stream = Files.list(src)) {
      stream.filter(p -> p.toString().endsWith(".rbbi")).sorted().forEach(files::add);
    }
    for (Path p : files) {
      String name = p.getFileName().toString().replace(".rbbi", "");
      String rules = Files.readString(p, StandardCharsets.UTF_8);
      Files.writeString(out.resolve(name + ".rbbi"), rules, StandardCharsets.UTF_8);
      run(out, name, rules, texts);
    }
    for (var e : inline().entrySet()) {
      Files.writeString(out.resolve("inline_" + e.getKey() + ".rbbi"), e.getValue(), StandardCharsets.UTF_8);
      run(out, "inline_" + e.getKey(), e.getValue(), texts);
    }
    // character names, as \N{name} looks them up
    List<String> probes = new ArrayList<>();
    for (int cp = 0; cp <= 0x10ffff; cp += (cp < 0x3000 ? 3 : cp < 0x20000 ? 53 : 211)) {
      String n = com.ibm.icu.lang.UCharacter.getName(cp);
      if (n != null && !n.isEmpty()) probes.add(n);
      String a = com.ibm.icu.lang.UCharacter.getNameAlias(cp);
      if (a != null && !a.isEmpty()) probes.add(a);
    }
    for (int cp = 0xac00; cp <= 0xd7a3; cp += 97) probes.add(com.ibm.icu.lang.UCharacter.getName(cp));
    for (int cp : new int[] {0x4e00, 0x9fff, 0x3400, 0x20000, 0x2a6df, 0x2a700, 0x2b739, 0x2b740,
        0x2b81d, 0x2b820, 0x2cea1, 0x2ceb0, 0x2ebe0, 0x30000, 0x31350, 0x323af, 0xf900, 0xfa0d,
        0x2f800, 0x17000, 0x187f7, 0x1b170, 0x18b00, 0x18cd5}) {
      String n = com.ibm.icu.lang.UCharacter.getName(cp);
      if (n != null) probes.add(n);
    }
    Collections.addAll(probes, "latin small letter a", "Latin Small Letter A", "LATIN  SMALL LETTER A",
        " LATIN SMALL LETTER A", "HYPHEN-MINUS", "hyphen-minus", "NBSP", "NULL", "null", "LINE FEED",
        "LF", "<control-0000>", "<control-0007>", "<control-001F>", "<control-007f>", "<control-0041>",
        "<private use-E000>", "<noncharacter-FFFE>", "<noncharacter-FDD0>", "<unassigned-0378>",
        "<surrogate-D800>", "<letter-0041>", "<control-zz>", "<control->", "<control-+7>", "<>", "<",
        "", "X", "NO SUCH CHARACTER", "CJK UNIFIED IDEOGRAPH-4E00", "CJK UNIFIED IDEOGRAPH-4e00",
        "CJK UNIFIED IDEOGRAPH-04E00", "CJK UNIFIED IDEOGRAPH-4DFF", "CJK UNIFIED IDEOGRAPH-",
        "HANGUL SYLLABLE GA", "HANGUL SYLLABLE GAG", "HANGUL SYLLABLE GAX", "HANGUL SYLLABLE HIH",
        "HANGUL SYLLABLE", "HANGUL SYLLABLE ", "hangul syllable ga");
    StringBuilder np = new StringBuilder();
    for (String n : probes) {
      np.append(AnalysisRows.esc(n)).append('\t')
          .append(com.ibm.icu.lang.UCharacter.getCharFromExtendedName(n)).append('\n');
    }
    Files.writeString(out.resolve("names_probe.tsv"), np.toString(), StandardCharsets.UTF_8);
    // ICUTokenizerFactory's rulefiles, through a CustomAnalyzer reading the rule files' directory
    String[] specs = {
      "icu,rulefiles=Latn:Latin-break-only-on-whitespace.rbbi",
      "icu,rulefiles=Latn:KeywordTokenizer.rbbi",
      "icu,rulefiles=Latn:Default.rbbi",
      "icu,rulefiles=Latn:Latin-dont-break-on-hyphens.rbbi",
      "icu,rulefiles=Latn:KeywordTokenizer.rbbi_COMMA_Cyrl:Latin-break-only-on-whitespace.rbbi",
      "icu,rulefiles=Hani:KeywordTokenizer.rbbi",
      "icu,rulefiles=Latn:Latin-break-only-on-whitespace.rbbi,cjkAsWords=false",
      "icu,rulefiles=Grek:MyanmarSyllable.rbbi",
      "icu,rulefiles=Latn:nope.rbbi",
      "icu,rulefiles=Latn:Bad.rbbi",
      "icu,rulefiles=Latn",
      "icu,rulefiles=Nope:KeywordTokenizer.rbbi",
    };
    List<String> corpus = AnalysisRows.corpus("analysis-icu.txt");
    StringBuilder sp = new StringBuilder();
    for (int i = 0; i < specs.length; i++) {
      String name = String.format("rulefiles_%02d", i);
      sp.append(name).append('\t').append(specs[i]).append('\n');
      String rows;
      try {
        String[] nameArgs = specs[i].split(",");
        List<String> argv = new ArrayList<>();
        for (int k = 1; k < nameArgs.length; k++) {
          String[] kv = nameArgs[k].split("=", 2);
          argv.add(kv[0]);
          argv.add(kv[1].replace("_COMMA_", ","));
        }
        try (Analyzer a =
            CustomAnalyzer.builder(src).withTokenizer(nameArgs[0], argv.toArray(new String[0])).build()) {
          rows = AnalysisRows.rows(a, corpus);
        }
      } catch (Exception ex) {
        Throwable c = ex.getCause() != null ? ex.getCause() : ex;
        rows = "B\t" + c.getClass().getSimpleName() + "\n";
      }
      Files.writeString(out.resolve(name + ".tsv"), rows, StandardCharsets.UTF_8);
    }
    Files.writeString(out.resolve("rulefiles_specs.tsv"), sp.toString(), StandardCharsets.UTF_8);
  }
}
