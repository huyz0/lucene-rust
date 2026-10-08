import com.ibm.icu.text.BreakIterator;
import com.ibm.icu.text.RuleBasedBreakIterator;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * M12 T12.4: ICU4J 77.1's {@code RuleBasedBreakIterator} over ICU's own character, line, sentence
 * and title rules ({@code getInstanceFromCompiledRules} on the jar's {@code .brk} files, copied here
 * byte for byte as {@code *.brk}), compared by {@code
 * crates/lucene-analysis-icu/tests/icu_breaks_fixtures.rs}. The word rules are {@code
 * GenAnalysisIcu}'s ({@code ICUTokenizer}); these reach the iterator paths the word rules do not:
 * start-of-text and lookahead rules, rule-status vectors, dictionary runs inside line breaking.
 *
 * <p>{@code breaks.txt}: one line per text (the analysis-icu corpus, 200 seeded strings), UTF-16
 * units in hex. {@code <name>.tsv}: per text, the boundaries {@code first()}..{@code next()} as
 * {@code position:ruleStatus:statusVector}.
 */
public class GenAnalysisIcuBreaks {
  static final String[] RULES = {"char", "line", "sent", "title", "line_loose_cj", "sent_el"};

  static final int[] POOL = {
    'a', 'B', ' ', ' ', '.', '?', '!', '"', '(', ')', '-', '1', '9', ',', '\n', '\r', 0x2029,
    0x3002, 0x3001, 0x300c, 0x300d, 0x3042, 0x30a2, 0x30fc, 0x4e00, 0x4e2d, 0xac00, 0x1100, 0x1161,
    0x11a8, 0xe01, 0xe32, 0xe40, 0xe48, 0x0e2a, 0x627, 0x5d0, 0x301, 0x200d, 0x1f600, 0x1f1fa,
    0x1f1f8, 0x1f3fb, 0xfe0f, 0xa0, 0x2014, 0x2026, 0x00ab, 0x00bb, 0x3b1, 0x37e, 0x2018, 0x2019,
    0x915, 0x94d, 0x937, 0x1000, 0x103a
  };

  static String hex(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      if (i > 0) b.append(' ');
      b.append(String.format("%04x", (int) s.charAt(i)));
    }
    return b.toString();
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_icu_breaks");
    Files.createDirectories(out);
    List<String> texts = new ArrayList<>(AnalysisRows.corpus("analysis-icu.txt"));
    Collections.addAll(texts, "", "a", " ", "Mr. Smith went to Washington. He said \"Hi!\" Then left.",
        "(Hello.) [World?] 'Quote.' 3.14 is pi. e.g. this. Etc. Next para.",
        "ภาษาไทยไม่มีการเว้นวรรค ฉันชอบกินข้าว", "日本語の文章です。次の文！", "Ἀθῆναι; τί; Ναι.",
        "กรุงเทพฯ ฯลฯ ไปมาฯ ดีๆ เด็กๆ ฯพณฯ", "ラーメンとひらがな漢字カタカナのテキスト", "ລາວຯ ພາສາລາວ",
        "ភាសាខ្មែរ", "မြန်မာဘာသာ", "ｶﾀｶﾅﾃｷｽﾄ と ひらがな");
    Random rnd = new Random(0x62726b73L);
    for (int i = 0; i < 200; i++) {
      StringBuilder sb = new StringBuilder();
      int len = 1 + rnd.nextInt(24);
      for (int j = 0; j < len; j++) sb.appendCodePoint(POOL[rnd.nextInt(POOL.length)]);
      texts.add(sb.toString());
    }
    StringBuilder t = new StringBuilder();
    for (String s : texts) t.append(hex(s)).append('\n');
    Files.writeString(out.resolve("breaks.txt"), t.toString(), StandardCharsets.UTF_8);
    for (String name : RULES) {
      byte[] brk;
      try (InputStream in =
          RuleBasedBreakIterator.class.getResourceAsStream(
              "/com/ibm/icu/impl/data/icudata/brkitr/" + name + ".brk")) {
        brk = in.readAllBytes();
      }
      Files.write(out.resolve(name + ".brk"), brk);
      RuleBasedBreakIterator bi = RuleBasedBreakIterator.getInstanceFromCompiledRules(ByteBuffer.wrap(brk));
      StringBuilder sb = new StringBuilder();
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
  }
}
