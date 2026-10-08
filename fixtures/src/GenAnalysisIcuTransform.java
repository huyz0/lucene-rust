import com.ibm.icu.text.Transliterator;
import com.ibm.icu.text.UnicodeSet;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.KeywordTokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.icu.ICUTransformFilter;

/**
 * M12 T12.4: ICU4J 77.1's transliterators ({@code Transliterator.getInstance}, {@code
 * createFromRules}, {@code transliterate}, {@code getSourceSet}) and Lucene's {@code
 * ICUTransformFilter}, compared by {@code crates/lucene-analysis-icu/tests/icu_transform_fixtures.rs}.
 *
 * <ul>
 *   <li>{@code tr_strings.txt}: the texts, one per line as space-separated UTF-16 units in hex: the
 *       analysis-icu corpus, curated strings, and 400 seeded strings from pools of letters of many
 *       scripts, marks, digits and punctuation.
 *   <li>{@code tr_ids.tsv}: one row per transliterator spec ({@code ID} or {@code ID|R} for the
 *       reverse direction) -- {@code spec getID fnv...}: FNV-1a 64 of each block of 50 outputs
 *       (each output's units then 0xFFFF), or {@code spec X Exception} for a refused one. The specs:
 *       every ID {@code getAvailableIDs()} lists, then compound, filtered, inverse-syntax, alias and
 *       malformed IDs.
 *   <li>{@code tr_full.tsv}: {@code spec line output} (output in hex units) of every string, for the
 *       transliterators Lucene's tests use.
 *   <li>{@code tr_sources.tsv}: {@code spec ranges} -- {@code getSourceSet()} as hex ranges, for the
 *       rule-based transliterators of {@code tr_full.tsv}.
 *   <li>{@code tr_rules.tsv}: {@code R index dir rules} (rules escaped as AnalysisRows does) then
 *       {@code O index line output} per rule-input string ({@code !Exception} for a
 *       transliteration that throws), or {@code X index Exception} for rules that do not parse.
 *   <li>{@code tr_lucene_*.tsv}: {@code ICUTransformFilter} over the corpus (AnalysisRows rows).
 * </ul>
 */
public class GenAnalysisIcuTransform {
  static final int BLOCK = 50;

  static final int[][] POOLS = {
    // Latin with accents, digraphs and case pairs
    {'a', 'b', 'c', 'e', 'h', 'i', 'k', 'n', 'o', 's', 't', 'u', 'y', 'z', 'A', 'E', 'I', 'O', 'S', 'T',
      0xe9, 0xe8, 0xf1, 0xfc, 0xdf, 0x131, 0x130, 0x1c6, 0x1c5, 0x149, 0x1e9e, 0xc6, 0x153, 0x142},
    // Cyrillic, Greek (final sigma), Armenian
    {0x430, 0x431, 0x436, 0x449, 0x44a, 0x44c, 0x451, 0x454, 0x457, 0x491, 0x410, 0x416, 0x429, 0x401,
      0x3b1, 0x3b2, 0x3c3, 0x3c2, 0x3a3, 0x3ac, 0x390, 0x3b0, 0x1f00, 0x1f80, 0x1fb3, 0x587, 0x561, 0x535},
    // Kana (small, voiced, long vowel), fullwidth and halfwidth forms
    {0x3042, 0x3041, 0x304b, 0x304c, 0x3063, 0x3083, 0x3093, 0x30a2, 0x30ab, 0x30ac, 0x30c3, 0x30e3,
      0x30f3, 0x30fc, 0x30f4, 0x309b, 0x3099, 0xff71, 0xff76, 0xff9e, 0xff21, 0xff41, 0xff10, 0xff01, 0x3000},
    // Han (traditional and simplified), Hangul syllables and jamo
    {0x4e2d, 0x570b, 0x56fd, 0x8a9e, 0x8bed, 0x9ad4, 0x4f53, 0x6f22, 0x6c49, 0x5b57, 0xac00, 0xd55c,
      0xae00, 0x1100, 0x1161, 0x11a8, 0x3131},
    // Indic: Devanagari, Bengali, Tamil, Thai, Arabic, Hebrew with marks
    {0x915, 0x92e, 0x93f, 0x94d, 0x93c, 0x902, 0x966, 0x995, 0x9be, 0x9cd, 0xb95, 0xbcd, 0xbca, 0xe01,
      0xe32, 0xe40, 0xe48, 0x627, 0x644, 0x628, 0x64e, 0x651, 0x5e9, 0x5c1, 0x5bc, 0x5b8},
    // marks, digits, spaces, punctuation, symbols, supplementary
    {0x300, 0x301, 0x308, 0x327, ' ', ' ', '-', '\'', '.', ',', '0', '7', 0x2019, 0x201c, 0xa0, 0x2028,
      0x1d400, 0x1f600, 0x10400, 0x20000, 0xfb01, 0x2163, 0x2460},
  };

  static final String[] CURATED = {
    "", "a", "A", "ΑΣ ΟΔΟΣ σ", "Ꭰ", "ǅungla Ǆ ǆ", "İIıi", "ﬃ ﬀ ﬁx", "ŉ", "ẞß",
    "ﾊﾞｶﾞ ｶﾞｷﾞ ｳﾞ ﾟ", "ＡＢＣ　１２３！", "ヴァイオリン ゔ ゎ ヵ", "キャッチャー ちゃっ", "かな カナ ﾊﾟﾝ",
    "中國語 中国语 漢字 汉字 發 发", "한국어 한글 ㄱㄴ", "Москва Щёкино Йошкар-Ола ЪЬ",
    "ελληνικά Αθήνα ψυχή", "مرحبا بالعالم", "שלום עולם", "नमस्ते क़लम", "தமிழ்", "ภาษาไทย",
    "ქართული", "Հայաստան", "ኢትዮጵያ", "ᏣᎳᎩ", "ᠮᠣᠩᠭᠣᠯ", "ꦗꦮ", "𐐀𐐨", "à́ ë",
    "xͅΙ", "'a'b''", "çà", "1/2 ½ ²", "  　", "Ǳ ǲ ǳ",
  };

  static final String[] EXTRA_SPECS = {
    "Any-Latin; Latin-ASCII", "NFD; [:Nonspacing Mark:] Remove; NFC", "[:Nonspacing Mark:] Remove",
    "[:Latin:] Lower", "[a-m] Upper", "Han-Latin/Names", "Any-Upper", "Any-Lower", "Any-Title",
    "Any-CaseFold", "Null", "Lower", "Upper", "Title", "CaseFold", "NFC", "NFD", "NFKC", "NFKD", "FCD",
    "FCC", "Any-NFD", "Remove", "Any-Remove", "Any-Null", "Any-Hex", "Hex-Any", "Any-Hex/XML",
    "Any-Name", "Name-Any", "Any-BreakInternal", "Latn-Cyrl", "Cyrl-Latn", "Latin-Katakana",
    "Katakana-Latin", "Hiragana-Katakana", "Katakana-Hiragana", "Traditional-Simplified",
    "Simplified-Traditional", "Fullwidth-Halfwidth", "Halfwidth-Fullwidth", "Han-Latin",
    "Cyrillic-Latin", "Latin-Cyrillic", "Greek-Latin", "Greek-Latin/UNGEGN", "Thai-Latin",
    "Any-Latin", "Any-Greek", "Any-Hangul", "Any-Katakana", "Any-Han", "Latin-Han", "Any-Devanagari",
    "el-Latin", "ja-Latin", "Latin-el", "Any-am_FONIPA", "Lower(Upper)", "NFD(NFC)",
    "(Any-Latin)", "([:Latin:] Upper)", "[:Greek:] Any-Latin; ::Lower", "::Lower;", "Any-Latin;",
    ";Any-Latin", "Any-Latin;;NFD", "[[:L:]-[a]]; Upper", "[^a-z]; Any-Upper; ([b])", "Any-Lower; ([A])",
    "Bogus-ID", "Latin-Bogus", "[a-z", "", " ", "Any-", "-Latin", "Any-Latin/", "Latin-ASCII|R",
    "Any-Latin|R", "Latin-Cyrillic|R", "Null|R", "Any-Upper|R", "Lower|R", "NFC|R",
    "NFD; [:Nonspacing Mark:] Remove; NFC|R", "Han-Latin|R", "Hex-Any|R",
  };

  static final String[] FULL_SPECS = {
    "Traditional-Simplified", "Simplified-Traditional", "Katakana-Hiragana", "Hiragana-Katakana",
    "Fullwidth-Halfwidth", "Halfwidth-Fullwidth", "Any-Latin", "NFD; [:Nonspacing Mark:] Remove",
    "NFD; [:Nonspacing Mark:] Remove; NFC", "Han-Latin", "Any-CaseFold", "Any-Upper", "Any-Title",
    "Cyrillic-Latin", "Latin-Cyrillic", "Cyrillic-Latin|R", "Greek-Latin", "Latin-ASCII", "Null",
    "Any-Latin; Latin-ASCII", "Hangul-Latin", "Arabic-Latin", "Hebrew-Latin", "Devanagari-Latin",
    "Thai-Latin", "Any-BreakInternal",
  };

  static final String[] RULES = {
    // simple, context, cursor, quantifiers, anchors, segments, variables, functions, filters
    "a > b; c > d;",
    "ab > x; a > y; b > z;",
    "x { a } y > A; a > b;",
    "a > b | c; c > d;",
    "a > | @@ x; x > y;",
    "a+ > X; b* > Y; c? > Z;",
    "^a > S; a } $ > E;",
    "$vowel = [aeiou]; $vowel { b > B;",
    "([a-z]) ([0-9]) > $2 $1;",
    "(ab) > &Any-Upper($1);",
    "([a-z]+) > &Any-Upper($1) '-' $1;",
    ":: [a-c] ; a > x; b > y; z > Z;",
    ":: Lower; ab > X;",
    ":: Upper; :: Null; A > b;",
    "a <> b; c <> d;",
    "a < b;",
    "'a' > 'b c'; \\u0041 > \\U0001F600; '' > q;",
    "[:Lu:] > X; [^a-z] > '';",
    "[[:Greek:]&[:L:]] > g;",
    "$a = x; $b = $a $a; $b > Y;",
    "a > ; b > '';",
    "# comment\na > b; # tail\n",
    "use variable range 0xE000 0xE0FF; a > b;",
    ":: NFD; [:Mn:] > ; :: NFC;",
    ":: [a-z] Upper (Lower) ;",
    "a } [:L:]* $ > X;",
    "[:L:] { b } > c; b > B;",
    "ab > xy | a; a > Q;",
    "abc > |@x; x > Y;",
    ":: Any-Latin ; :: Latin-ASCII ;",
    "$x = [abc]; ($x+) > \\{ $1 \\};",
    "\\u0061 > \\u00e9 ;",
    // malformed
    "a > b",
    "a >",
    "$undefined > x;",
    "(a > b;",
    "a } } b > c;",
    "[a-z > x;",
    ":: Bogus-ID ;",
    "a > $1;",
    "use variable range 0xE000 0xE001; $a=a; $b=b; $c=c; $d=d; x > y;",
    "a > b; ::Lower; c > d;",
    ":: [a-z] ; :: [b-c] ; a > b;",
    ":: Bogus ID ;",
    "use foo;",
    "use maximum backup 1; a > b;",
    "use nfd rules; a > b;",
    "use nfc rules; a > b;",
    "$a = ;",
    "$ = x;",
    "$a b = c;",
    "a b c",
    "a ;",
    "a > b > c;",
    "| a > b;",
    "a | b > c;",
    "$1 > x;",
    "$ > x;",
    "a\\",
    "\\u12 > x;",
    "'abc > x;",
    "'a''b' > x; '' > q;",
    "a ^ b > x;",
    "&Bogus(a) > x;",
    "a > &Any-Upper(;",
    "a $ > x;",
    "a $ b > c;",
    "* > x;",
    "'ab'+ > x;",
    "a { b { c > x;",
    "a > b | c | d;",
    "a > @b;",
    "a > b@@;",
    "a > |@@b;",
    "a > @@|b;",
    "a } b } c > x;",
    "use variable range 0xE000 0xE000; $a=[a]; $b=[b]; x > y;",
    "::[a-z] Lower; ::Upper;",
    "a > b; :: [a-z];",
    "a > b; ::([a-z]);",
    "a b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;\na b;",
    "$a = [a-z]; $a > x; $a = [b];",
    "([a-z]) ([0-9]) > $3;",
    "(a (b)) > $2 $1;",
    "{a} > x;",
    "a > b ; # comment without newline",
    "\\u0041 > ;",
    ". > x;",
    "a <> b > c;",
    "a \u2190 b; c \u2192 d; e \u2194 f;",
    "$a = ''; $a a > x;",
    "[:Lu:] [:Ll:] > x | y;",
    "a } [b-z]+ $ > X;",
    "^(a) > b;",
    "([:L:]+) > &Any-Upper($1) &Any-Lower($1);",
    "a > &Any-Upper(b);",
    "\\$ > x; \\\\ > y;",
    "a < b > c;",
    "a > b;;;",
    "::Null;",
    ":: NFD (NFC) ;",
    "::Lower; ::Upper(Lower);",
    "a{b}c > x; ab > y;",
    "x { a > y;",
    "a } x > y;",
    "$v = a b; $v > c;",
    "$v = [abc]; $w = $v $v; ($w) > |$1;",
    "a > \\u0062 \\U0001F600;",
    "a+ > b; a > c;",
    "a > b; a > c;",
    "[a] > b;\n[a] > c;",
    "'a' > 'b';\n'a' > 'c';",
    "use variable range 0x41 0x42; a > b;",
    "use variable range 0xE000; a > b;",
    "$a = x; $a = y;",
    "$Var = x; $var > y;",
    "a > b # c\n; d > e;",
    ":: Any-Latin ( Latin-Any ) ;",
    "use variable range 0xE000 0x7FFFFFFFFF; a > b;",
    "use variable range 0160000 0160377; a > b;",
    "use variable range 0x 0xE0FF; a > b;",
    "use variable range0xE000 0xE0FF; a > b;",
    "use variable range x 0xE0FF; a > b;",
    "use variable rang 1 2; a > b;",
    "use variable range 0x10 0x5; a > b;",
    "use variable range 0xE000 0xE001; $a = [a]; $b = [b]; $c = [c]; x > y;",
    "use variable range 0xE000 0xE000; a+ > x;",
    "use variable range 0xE000 0xE001; (a)(b)(c) > x;",
    "$x $y > z;",
    "use variable range 0xE000 0xE000; $q = [a]; $x > z;",
    "$a = ^b;",
    "$a = b $;",
    "use variable range 0x61 0x7a; a > b;",
    "use variable range 0x61 0x7a; 'a' > b;",
    "(*a) > x;",
    "a > @@b;",
    "a > b@|c;",
    "a > |b@;",
    "a > @b@;",
    "a > b@c@;",
    "a > b$",
    "a > b; c$",
    "$1x > x;",
    "&[a-z] Any-Upper(a) > x;",
    "&Any-Upper a > b;",
    "(a|b) > c;",
    "(a{b) > c;",
    "[a > b;",
    "a > [b];",
    "x > (a);",
    "(a) [$1] > x;",
    "$a = [a-z]; [$a-[q]] > x;",
    "a > \\u12;",
    "a } [:L:] $ > x; a > y;",
    "^a > x; ^ab > y;",
    "a ^ > b;",
    "^^a > b;",
    "a > b c | d @ @;",
    "\u2206Any-Upper(a) > x;",
    "a > &Any-Upper(\u2206Any-Lower(b));",    "",
    "# nothing\n",
    ":: [a-z] ; :: Upper ;",
    ":: [a-z] ; :: Upper ; :: Lower ;",
    "::Upper; a > b; ::Lower; b > c;",
    ":: [a-z] ; a > b; :: Lower ;",
    ":: Null ;",
    "::Any-Null;",
  };

  static final String[] RULE_INPUTS = {
    "", "a", "ab", "abc", "aab", "ba", "xay", "aaaa", "bbb", "c", "cab", "a1", "b2c3", "Hello World",
    "ABC abc", "zzz", "é", "é", "αβγ", "Москва", "x a y", "aXb", "123", "a b c", "aa bb cc",
  };

  static String unitsHex(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      if (i > 0) b.append(' ');
      b.append(String.format("%04x", (int) s.charAt(i)));
    }
    return b.toString();
  }

  static long fnv(long h, String s) {
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      h ^= c & 0xff;
      h *= 0x100000001b3L;
      h ^= c >>> 8;
      h *= 0x100000001b3L;
    }
    h ^= 0xff;
    h *= 0x100000001b3L;
    h ^= 0xff;
    h *= 0x100000001b3L;
    return h;
  }

  static Transliterator instance(String spec) {
    int bar = spec.lastIndexOf('|');
    if (bar >= 0 && spec.substring(bar).equals("|R")) {
      return Transliterator.getInstance(spec.substring(0, bar), Transliterator.REVERSE);
    }
    return Transliterator.getInstance(spec);
  }

  static String ranges(UnicodeSet set) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < set.getRangeCount(); i++) {
      if (i > 0) b.append(',');
      b.append(Integer.toHexString(set.getRangeStart(i))).append('-').append(Integer.toHexString(set.getRangeEnd(i)));
    }
    for (String s : set.strings()) b.append(",{").append(unitsHex(s)).append('}');
    return b.toString();
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_icu_transform");
    Files.createDirectories(out);
    List<String> corpus = AnalysisRows.corpus("analysis-icu.txt");
    List<String> texts = new ArrayList<>(corpus);
    Collections.addAll(texts, CURATED);
    Random rnd = new Random(0x7472616eL);
    for (int i = 0; i < 400; i++) {
      int[] pool = POOLS[rnd.nextInt(POOLS.length)];
      int[] other = POOLS[rnd.nextInt(POOLS.length)];
      int len = 1 + rnd.nextInt(12);
      StringBuilder sb = new StringBuilder();
      for (int j = 0; j < len; j++) {
        int[] p = rnd.nextInt(4) == 0 ? other : pool;
        sb.appendCodePoint(p[rnd.nextInt(p.length)]);
      }
      texts.add(sb.toString());
    }
    StringBuilder ts = new StringBuilder();
    for (String t : texts) ts.append(unitsHex(t)).append('\n');
    Files.writeString(out.resolve("tr_strings.txt"), ts.toString(), StandardCharsets.UTF_8);

    List<String> specs = new ArrayList<>();
    for (java.util.Enumeration<String> e = Transliterator.getAvailableIDs(); e.hasMoreElements(); ) specs.add(e.nextElement());
    Collections.sort(specs);
    Collections.addAll(specs, EXTRA_SPECS);
    StringBuilder ids = new StringBuilder();
    for (String spec : specs) {
      ids.append(spec);
      try {
        Transliterator t = instance(spec);
        StringBuilder row = new StringBuilder().append('\t').append(t.getID());
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < texts.size(); i++) {
          h = fnv(h, t.transliterate(texts.get(i)));
          if ((i + 1) % BLOCK == 0 || i + 1 == texts.size()) {
            row.append('\t').append(Long.toHexString(h));
            h = 0xcbf29ce484222325L;
          }
        }
        ids.append(row);
      } catch (RuntimeException ex) {
        ids.append("\tX\t").append(ex.getClass().getSimpleName());
      }
      ids.append('\n');
    }
    Files.writeString(out.resolve("tr_ids.tsv"), ids.toString(), StandardCharsets.UTF_8);

    StringBuilder full = new StringBuilder();
    StringBuilder sources = new StringBuilder();
    for (String spec : FULL_SPECS) {
      Transliterator t = instance(spec);
      for (int i = 0; i < texts.size(); i++) {
        full.append(spec).append('\t').append(i).append('\t').append(unitsHex(t.transliterate(texts.get(i)))).append('\n');
      }
      if (t.getClass().getSimpleName().equals("RuleBasedTransliterator")) {
        sources.append(spec).append('\t').append(ranges(t.getSourceSet())).append('\n');
      }
    }
    Files.writeString(out.resolve("tr_full.tsv"), full.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("tr_sources.tsv"), sources.toString(), StandardCharsets.UTF_8);

    StringBuilder rules = new StringBuilder();
    for (int r = 0; r < RULES.length; r++) {
      for (int dir = 0; dir < 2; dir++) {
        String key = r + (dir == 0 ? "F" : "R");
        rules.append("R\t").append(key).append('\t').append(AnalysisRows.esc(RULES[r])).append('\n');
        try {
          Transliterator t = Transliterator.createFromRules("Test", RULES[r], dir);
          rules.append("I\t").append(key).append('\t').append(t.getID()).append('\n');
          for (int i = 0; i < RULE_INPUTS.length; i++) {
            String output;
            try {
              output = unitsHex(t.transliterate(RULE_INPUTS[i]));
            } catch (RuntimeException ex) {
              output = "!" + ex.getClass().getSimpleName();
            }
            rules.append("O\t").append(key).append('\t').append(i).append('\t').append(output).append('\n');
          }
        } catch (RuntimeException ex) {
          rules.append("X\t").append(key).append('\t').append(ex.getClass().getSimpleName()).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("tr_rules.tsv"), rules.toString(), StandardCharsets.UTF_8);

    Map<String, Supplier<Analyzer>> chains = new LinkedHashMap<>();
    String[][] lucene = {
      {"tr_lucene_ws_any_latin", "Any-Latin", "F", "ws"},
      {"tr_lucene_ws_trad_simp", "Traditional-Simplified", "F", "ws"},
      {"tr_lucene_ws_kata_hira", "Katakana-Hiragana", "F", "ws"},
      {"tr_lucene_ws_full_half", "Fullwidth-Halfwidth", "F", "ws"},
      {"tr_lucene_ws_strip_marks", "NFD; [:Nonspacing Mark:] Remove", "F", "ws"},
      {"tr_lucene_ws_han_latin", "Han-Latin", "F", "ws"},
      {"tr_lucene_ws_casefold", "CaseFold", "F", "ws"},
      {"tr_lucene_ws_cyrl_latn", "Cyrillic-Latin", "F", "ws"},
      {"tr_lucene_ws_latn_cyrl_rev", "Cyrillic-Latin", "R", "ws"},
      {"tr_lucene_ws_null", "Null", "F", "ws"},
      {"tr_lucene_kw_any_latin_ascii", "Any-Latin; Latin-ASCII; Lower", "F", "kw"},
    };
    for (String[] l : lucene) {
      String id = l[1];
      int dir = l[2].equals("R") ? Transliterator.REVERSE : Transliterator.FORWARD;
      boolean ws = l[3].equals("ws");
      chains.put(l[0], () -> AnalysisRows.chain(
          ws ? WhitespaceTokenizer::new : KeywordTokenizer::new,
          t -> new ICUTransformFilter(t, Transliterator.getInstance(id, dir))));
    }
    chains.put("tr_lucene_ws_rules", () -> AnalysisRows.chain(
        WhitespaceTokenizer::new,
        t -> new ICUTransformFilter(t, Transliterator.createFromRules("test", "a > b; [:Lu:] > X; (é) > &Any-Upper($1);", Transliterator.FORWARD))));
    AnalysisRows.writeChains(out, chains, corpus);
  }
}
