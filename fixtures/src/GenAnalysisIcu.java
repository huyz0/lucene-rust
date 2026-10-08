import com.ibm.icu.lang.UCharacter;
import com.ibm.icu.text.Normalizer;
import com.ibm.icu.text.Normalizer2;
import com.ibm.icu.text.UnicodeSet;
import java.io.InputStream;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.KeywordTokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.icu.ICUFoldingFilter;
import org.apache.lucene.analysis.icu.ICUNormalizer2CharFilter;
import org.apache.lucene.analysis.icu.ICUNormalizer2Filter;
import org.apache.lucene.analysis.icu.IcuAccess;
import org.apache.lucene.analysis.icu.segmentation.DefaultICUTokenizerConfig;
import org.apache.lucene.analysis.icu.segmentation.ICUTokenizer;
import org.apache.lucene.analysis.icu.tokenattributes.ScriptAttribute;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;

/**
 * M12 T12.4: analysis-icu over ICU4J 77.1 ({@code crates/lucene-analysis-icu/tests/icu_fixtures.rs}
 * compares).
 *
 * <ul>
 *   <li>{@code norm_blocks.tsv}: every 256-code-point block holding an assigned character (surrogates
 *       left out), as one string, through six normalizers ({@code nfc nfkc nfkc_cf nfkc_scf uts46}
 *       and Lucene's {@code utr30}) in four modes: {@code form mode start fnv(normalize) quickCheck
 *       isNormalized spanQuickCheckYes fnv(per code point hasBoundaryBefore, hasBoundaryAfter,
 *       isInert)}, the digests FNV-1a 64 over UTF-16 units.
 *   <li>{@code norm_strings.txt}: 600 seeded strings from pools that stress normalization (combining
 *       marks of every class in every order, Hangul jamo and syllables, compatibility forms, case
 *       folding, default ignorables, lone surrogates); {@code norm_strings.tsv}: each through every
 *       normalizer and mode -- {@code form mode index normalize quickCheck spanQuickCheckYes
 *       normalizeSecondAndAppend(normalize(first part), second part)}.
 *   <li>{@code unicode_sets.tsv}: {@code UnicodeSet} patterns -- curated, every General_Category and
 *       Script value, every binary property -- as {@code pattern size ranges fnv(ranges)} or {@code
 *       pattern X Exception}.
 *   <li>{@code tok_*.tsv}: {@code ICUTokenizer} in its four configurations ({@code cjkAsWords},
 *       {@code myanmarAsWords}) over the corpus and {@code tok_stress.txt} (300 seeded lines from
 *       pools of every script the break rules and dictionaries treat apart -- Thai, Lao, Khmer,
 *       Myanmar, Tai Tham, Han, kana, Hangul, emoji sequences, keycaps, digits, marks -- some past
 *       the 4,096-unit buffer, with and without white space to cut at): {@code T line term start
 *       end posInc type scriptCode scriptReflected}, {@code E line finalStart finalEnd}.
 *   <li>chains ({@link AnalysisRows}): the filters and char filter over {@code
 *       corpus/analysis-icu.txt} (written here, apart from short public-domain quotations) and the
 *       strings, the char filter with 2-, 3- and 5-unit buffers, and the factories through {@code
 *       CustomAnalyzer} (with refused configurations).
 * </ul>
 */
public class GenAnalysisIcu {
  static final String[] FORMS = {"nfc", "nfkc", "nfkc_cf", "nfkc_scf", "uts46", "utr30"};
  static final String[] MODES = {"compose", "decompose", "fcd", "fcc"};

  static Normalizer2 normalizer(String form, String mode) throws Exception {
    Normalizer2.Mode m =
        switch (mode) {
          case "compose" -> Normalizer2.Mode.COMPOSE;
          case "decompose" -> Normalizer2.Mode.DECOMPOSE;
          case "fcd" -> Normalizer2.Mode.FCD;
          default -> Normalizer2.Mode.COMPOSE_CONTIGUOUS;
        };
    if (form.equals("utr30")) {
      try (InputStream in = ICUFoldingFilter.class.getResourceAsStream("utr30.nrm")) {
        return Normalizer2.getInstance(in, "utr30", m);
      }
    }
    return Normalizer2.getInstance(null, form, m);
  }

  static long fnv(CharSequence s) {
    long h = 0xcbf29ce484222325L;
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      h = (h ^ (c >> 8)) * 0x100000001b3L;
      h = (h ^ (c & 0xff)) * 0x100000001b3L;
    }
    return h;
  }

  static String qc(Normalizer.QuickCheckResult r) {
    return r == Normalizer.YES ? "Y" : r == Normalizer.NO ? "N" : "M";
  }

  // --- the stress strings --------------------------------------------------

  static final int[][] POOLS = {
    // Latin bases, precomposed letters and case pairs
    {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'O', 's', 'S', 'k', 'K', 0xe9, 0xc5, 0x212b, 0x2126, 0x1e9b, 0x1e69,
      0xdf, 0x1e9e, 0x130, 0x131, 0x3a3, 0x3c2, 0x3c3, 0x149, 0x1f0, 0x390, 0x1e0b, 0x1e0d},
    // combining marks of many classes (1 overlay, 7 nukta, 202, 214, 220, 230, 232, 233, 234, 240)
    {0x300, 0x301, 0x302, 0x308, 0x30a, 0x323, 0x327, 0x328, 0x31b, 0x334, 0x335, 0x345, 0x35c, 0x35d,
      0x0e38, 0x0e48, 0x093c, 0x094d, 0x05b0, 0x05bc, 0x0f71, 0x0f72, 0x0f80, 0x0f74, 0x1dce, 0x20d2, 0x302a,
      0x3099, 0x309a, 0x1d165, 0x1d16d, 0x1d167},
    // Hangul jamo, syllables and compatibility jamo
    {0x1100, 0x1101, 0x1112, 0x1161, 0x1175, 0x11a7, 0x11a8, 0x11c2, 0x11c3, 0xac00, 0xac01, 0xd7a3, 0xd7a4,
      0x3131, 0x314f, 0xffa1, 0x115f, 0x1160},
    // compatibility forms
    {0xfb01, 0xfb03, 0x1c5, 0x1c4, 0x2163, 0x2460, 0x3251, 0x33a2, 0x337f, 0xff21, 0xff41, 0xff76, 0xff9e,
      0xb2, 0x2082, 0xbd, 0x2153, 0x1d400, 0x1d7ce, 0x2f00, 0xf900, 0x2f800, 0x3000, 0xfe64, 0xfdfa, 0xfb4f,
      0x1f130, 0x1f240, 0xa0, 0x2002, 0x2011, 0x2024},
    // default ignorables, controls, joiners and variation selectors
    {0xad, 0x34f, 0x200b, 0x200c, 0x200d, 0x200e, 0x2060, 0xfeff, 0xfe00, 0xfe0f, 0xe0100, 0xe0001, 0x180b,
      0x115f, 0x1160, 0x3164, 0xffa0, 0x1d173, 0x9, 0x85},
    // scripts with reordering or composition: Devanagari, Bengali, Tamil, Tibetan, Thai, Arabic, Hebrew
    {0x915, 0x93c, 0x958, 0x9c7, 0x9be, 0x9cb, 0x9d7, 0xbc6, 0xbbe, 0xbca, 0xbd7, 0xf40, 0xf71, 0xf73,
      0xf75, 0xf81, 0xe01, 0xe33, 0xe4d, 0x627, 0x653, 0x622, 0x5d0, 0x5b7, 0xfb2e, 0x1b05, 0x1b35, 0x1b06,
      0x11099, 0x110ba, 0x1109a},
    // lone surrogates, noncharacters, supplementary
    {0xd800, 0xdbff, 0xdc00, 0xdfff, 0xfffe, 0xffff, 0x1f600, 0x1f3fd, 0x10400, 0x20000, 0x10ffff, 0x1d15e,
      0x1d15f, 0x1d1bb},
  };

  static List<String> stress() {
    Random r = new Random(4242);
    List<String> out = new ArrayList<>();
    for (int i = 0; i < 600; i++) {
      StringBuilder b = new StringBuilder();
      int n = 1 + r.nextInt(i % 50 == 49 ? 200 : 12);
      for (int k = 0; k < n; k++) {
        int[] pool = POOLS[r.nextInt(POOLS.length)];
        b.appendCodePoint(pool[r.nextInt(pool.length)]);
      }
      // A string with a lone surrogate at its end may not be followed by a trail one.
      out.add(b.toString());
    }
    return out;
  }

  // --- UnicodeSet patterns -----------------------------------------------

  static List<String> unicodeSetPatterns() {
    List<String> p = new ArrayList<>(List.of(
        "[a-z]", "[^a-z]", "[-a]", "[^-a]", "[a-z-]", "[a\\-z]", "[\\u0041-\\u005A]", "[\\x41\\x{42}\\U00000043]",
        "[\\101]", "[\\t\\n\\r]", "[\\cA\\cZ]", "[ a b c ]", "  [abc]  ", "[]", "[^]", "[[a-z]-[aeiou]]",
        "[[a-z]&[c-x]]", "[[a-c][x-z]]", "[a[x]]", "[[a][b]-]", "[a$]", "[$a]", "[$x]", "[{abc}]", "[{a}]",
        "[{a}-{c}]", "[a-b{xy}c]", "[:Latin:]", "[:^Latin:]", "\\p{Lu}", "\\P{Lu}", "\\p{gc=Lu}",
        "\\p{General_Category=Letter}", "[\\p{sc=Grek}]", "[\\p{scx=Hani}]", "[:Nonspacing Mark:]",
        "[:White_Space:]", "[:White_Space=No:]", "[:lb=SA:]", "[:ccc=230:]", "[:ccc=Above:]", "[:ANY:]",
        "[:ascii:]", "[:Assigned:]", "[[:Emoji:][:Extended_Pictographic:]]", "[[:Thai:]&[:LineBreak=SA:]]",
        "[[:Thai:]&[:LineBreak=SA:]&[:M:]]", "[[:Mymr:]&[:LineBreak=SA:]]", "[\\u002a\\u00230-9©®™〰〽]",
        "[^åäöÅÄÖ]", "[[:L:]&[:^Latin:]]", "[\\p{Hangul_Syllable_Type=LV}]", "[:blk=Basic_Latin:]",
        "[:InBasicLatin:]", "[:ea=W:]", "[:wb=ALetter:]", "[:sb=STerm:]", "[:gcb=Extend:]", "[:nt=De:]",
        "[:bc=AL:]", "[:dt=Font:]", "[:jt=D:]", "[:vo=U:]", "[:InPC=Top:]", "[:InSC=Virama:]",
        "[:NFC_QC=M:]", "[:NFKD_QC=N:]", "[:lccc=230:]", "[:tccc=1:]", "[:bpt=o:]", "[:scx=Arab:]",
        "[:Script_Extensions=Deva:]", "[:Greek:]", "[:Zyyy:]", "[:Inherited:]", "[:Unknown:]", "[:Cn:]",
        "[:LC:]", "[:punct:]", "[:L&:]", "[: Lu :]", "[:^ Lu:]", "\\p{ lu }", "\\p{Lowercase}",
        "\\p{Lowercase=Yes}", "\\p{Lower=F}", "\\p{alpha}", "\\p{ID_Start}", "\\p{XID_Continue}",
        "\\p{Default_Ignorable_Code_Point}", "\\p{Emoji_Presentation}", "\\p{RGI_Emoji}",
        // refused by ICU4J too
        "a", "[a", "[a-]x", "[z-a]", "[a-[b]]", "[[a]&]", "[a&[b]]", "[^^]", "[{ab]", "[[a]-b]", "[{ab}-c]",
        "[a-{bc}]", "[[a]&{b}]", "[[:Latin:]&]", "\\p{Lu", "\\p{NoSuchThing}", "\\p{gc=Nope}", "\\p{Nope=x}",
        "[:Line_Break:]", "[:ccc=300:]", "[:ccc=x1:]", "[\\", "[\\u12]", "[\\x{12]", "[\\U00110000]", "[a] b",
        "[:L:", "\\px", "[[a]-$]", "\\p{lc=a}", "[:Age=x:]", "[:Age=x.1:]",
        "[:age=3.2:]", "[:Age= 1.1 :]", "[:age=16.0:]", "[:age=1.2.3.4:]", "[:age=1.2.3.4.5:]", "[:age=256:]",
        "\\p{nv=1}", "\\p{nv=0.5}", "\\p{nv=1e3}", "\\p{nv= 12 }", "\\p{nv=x}", "\\p{nv=1d}", "[:^lc:]",
        "[:Age:]"));
    for (int gc = 0; gc < 30; gc++) {
      String n = UCharacter.getPropertyValueName(com.ibm.icu.lang.UProperty.GENERAL_CATEGORY, gc, 0);
      if (n != null) {
        p.add("[:" + n + ":]");
        p.add("\\p{gc=" + UCharacter.getPropertyValueName(com.ibm.icu.lang.UProperty.GENERAL_CATEGORY, gc, 1) + "}");
      }
    }
    for (int sc = 0; sc < UCharacter.getIntPropertyMaxValue(com.ibm.icu.lang.UProperty.SCRIPT) + 1; sc++) {
      String n = UCharacter.getPropertyValueName(com.ibm.icu.lang.UProperty.SCRIPT, sc, 1);
      if (n != null) {
        p.add("[:" + n + ":]");
        p.add("\\p{scx=" + UCharacter.getPropertyValueName(com.ibm.icu.lang.UProperty.SCRIPT, sc, 0) + "}");
      }
    }
    for (int bp = com.ibm.icu.lang.UProperty.BINARY_START; bp < com.ibm.icu.lang.UProperty.BINARY_LIMIT; bp++) {
      String n = UCharacter.getPropertyName(bp, 1);
      if (n != null) p.add("[:" + n + ":]");
    }
    return p;
  }

  static String setRow(String pattern) {
    try {
      UnicodeSet s = new UnicodeSet(pattern);
      StringBuilder ranges = new StringBuilder();
      for (int i = 0; i < s.getRangeCount(); i++) {
        ranges.append(Integer.toHexString(s.getRangeStart(i))).append('-')
            .append(Integer.toHexString(s.getRangeEnd(i))).append(',');
      }
      for (String str : s.strings()) ranges.append('{').append(AnalysisRows.esc(str)).append('}');
      return AnalysisRows.esc(pattern) + "\t" + s.size() + "\t" + s.getRangeCount() + "\t"
          + Long.toHexString(fnv(ranges));
    } catch (Exception e) {
      return AnalysisRows.esc(pattern) + "\tX\t" + e.getClass().getSimpleName();
    }
  }

  // --- chains ------------------------------------------------------------

  static Supplier<Analyzer> factory(String spec) {
    return () -> {
      try {
        CustomAnalyzer.Builder b = CustomAnalyzer.builder();
        for (String part : spec.split(" ")) {
          String[] kv = part.split(":", 2);
          String[] nameArgs = kv[1].split(",");
          List<String> args = new ArrayList<>();
          for (int i = 1; i < nameArgs.length; i++) {
            String[] a = nameArgs[i].split("=", 2);
            args.add(a[0]);
            args.add(a[1].replace("_SPACE_", " ").replace("_COMMA_", ","));
          }
          String[] argv = args.toArray(new String[0]);
          switch (kv[0]) {
            case "c" -> b.addCharFilter(nameArgs[0], argv);
            case "t" -> b.withTokenizer(nameArgs[0], argv);
            default -> b.addTokenFilter(nameArgs[0], argv);
          }
        }
        return b.build();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    };
  }

  static final String[] SCRIPT_POOLS = {
    "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ๎ฯ๏๐๑๒",
    "ກຂຄງຈຊຍດຕຖທນບປຜຝພຟມຢຣລວສຫອຮຯະັາຳິີຶືຸູົຼຽເແໂໃໄໆ່້໊໋໌ໍໜໝ໐໑",
    "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឩឪឫឬឭឮឯឰឱឲឳាិីឹឺុូួើឿៀេែៃោៅំះៈ៉៊់៌៍៎៏័៑្៓។៕៖ៗ៘៙៚៛ៜ៝០១",
    "ကခဂဃငစဆဇဈဉညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအဢဣဤဥဦဧဨဩဪါာိီုူေဲဳဴဵံ့း္်ျြွှဿ၀၁၂၊။၌၍၎၏ၐၑၒၓၔၕၖၗ",
    "ᨠᨡᨢᨣᨤᨥᨦᨧᨨᨩᨪᨫᨬᨭᨮᨯᨰᨱᨲᨳᩕᩖᩗᩘᩙᩚᩛᩜᩝᩞ᩠ᩡᩢᩣᩤᩥᩦᩧᩨᩩᩪᩫᩬᩭᩮᩯ",
    "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可東京大学日本語勉強漢字𠀀𠀁",
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげごゝゞー",
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴヽヾーｱｲｳｴｵｰﾞﾟ",
    "가나다라마바사아자차카타파하각난닫랄맘밥삿앙잦찿칼탈팔할한글서울ㄱㄴㄷㅏㅓ",
  };

  static final String[] OTHER_TOKENS = {
    " ", " ", " ", "\t", "　", "hello", "World", "ICU77", "x86_64", "3.14", "1,000", "１２３", "ＡＢＣ", "e-mail",
    "don't", "😀", "👍🏽", "👨‍👩‍👧", "🇯🇵", "#️⃣", "1️⃣", "©️", "™", "❤️", "a\u0301", "\u0301", "\u200d", "\u200c",
    ".", ",", "。", "、", "!", "?", "-", "_", "@", "--", "'", "\"", "(", ")", "[", "]", "ﾊﾝｶｸ", "ー", "ｰ", "ﾞ",
  };

  static List<String> tokenizerStress() {
    Random r = new Random(7717);
    List<String> lines = new ArrayList<>();
    for (int i = 0; i < 300; i++) {
      StringBuilder b = new StringBuilder();
      int len = i % 30 == 29 ? 4100 + r.nextInt(5000) : 5 + r.nextInt(120);
      boolean noSpace = i % 60 == 59;
      while (b.length() < len) {
        if (r.nextInt(4) == 0) {
          String t = OTHER_TOKENS[r.nextInt(OTHER_TOKENS.length)];
          if (noSpace && t.isBlank()) continue;
          b.append(t);
        } else {
          String pool = SCRIPT_POOLS[r.nextInt(SCRIPT_POOLS.length)];
          int n = 1 + r.nextInt(12);
          for (int k = 0; k < n; k++) b.appendCodePoint(pool.codePointAt(pool.offsetByCodePoints(0, r.nextInt(pool.codePointCount(0, pool.length())))));
        }
      }
      lines.add(b.toString().replace("\n", " ").replace("\r", " "));
    }
    return lines;
  }

  static String tokRows(Analyzer a, List<String> lines) throws Exception {
    StringBuilder m = new StringBuilder();
    for (int ln = 0; ln < lines.size(); ln++) {
      try (TokenStream ts = a.tokenStream("f", lines.get(ln))) {
        CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
        OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);
        PositionIncrementAttribute inc = ts.addAttribute(PositionIncrementAttribute.class);
        TypeAttribute type = ts.addAttribute(TypeAttribute.class);
        ScriptAttribute script = ts.addAttribute(ScriptAttribute.class);
        ts.reset();
        while (ts.incrementToken()) {
          String[] reflected = {null};
          ((org.apache.lucene.util.AttributeImpl) script).reflectWith((c, k, v) -> reflected[0] = String.valueOf(v));
          m.append("T\t").append(ln).append('\t').append(AnalysisRows.esc(term.toString())).append('\t')
              .append(off.startOffset()).append('\t').append(off.endOffset()).append('\t')
              .append(inc.getPositionIncrement()).append('\t').append(type.type()).append('\t')
              .append(script.getCode()).append('\t').append(script.getShortName()).append('\t')
              .append(reflected[0]).append('\n');
        }
        ts.end();
        m.append("E\t").append(ln).append('\t').append(off.startOffset()).append('\t').append(off.endOffset())
            .append('\n');
      }
    }
    return m.toString();
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_icu");
    Files.createDirectories(out);

    // ICUTokenizer first, so the process-wide break engines are created in the order the Rust test
    // creates them (see crates/lucene-analysis-icu/src/icu4j/break_engines.rs).
    List<String> tokCorpus = AnalysisRows.corpus("analysis-icu.txt");
    List<String> tokStress = tokenizerStress();
    StringBuilder ts = new StringBuilder();
    for (String l : tokStress) ts.append(AnalysisRows.esc(l)).append('\n');
    Files.writeString(out.resolve("tok_stress.txt"), ts.toString(), StandardCharsets.UTF_8);
    List<String> tokLines = new ArrayList<>(tokCorpus);
    tokLines.addAll(tokStress);
    for (boolean cjk : new boolean[] {true, false}) {
      for (boolean myanmar : new boolean[] {true, false}) {
        Analyzer a = AnalysisRows.tok(() -> new ICUTokenizer(new DefaultICUTokenizerConfig(cjk, myanmar)));
        Files.writeString(out.resolve("tok_" + (cjk ? "cjk" : "nocjk") + "_" + (myanmar ? "mywords" : "mysyl") + ".tsv"),
            tokRows(a, tokLines), StandardCharsets.UTF_8);
        a.close();
      }
    }

    // Normalization over every assigned block.
    StringBuilder blocks = new StringBuilder();
    for (String form : FORMS) {
      for (String mode : MODES) {
        Normalizer2 n = normalizer(form, mode);
        for (int start = 0; start < 0x110000; start += 256) {
          StringBuilder s = new StringBuilder();
          StringBuilder bounds = new StringBuilder();
          boolean any = false;
          for (int c = start; c < start + 256; c++) {
            if (c >= 0xd800 && c <= 0xdfff) continue;
            if (UCharacter.getType(c) != UCharacter.UNASSIGNED) any = true;
            s.appendCodePoint(c);
            bounds.append(n.hasBoundaryBefore(c) ? '1' : '0').append(n.hasBoundaryAfter(c) ? '1' : '0')
                .append(n.isInert(c) ? '1' : '0');
          }
          if (!any) continue;
          blocks.append(form).append('\t').append(mode).append('\t').append(Integer.toHexString(start))
              .append('\t').append(Long.toHexString(fnv(n.normalize(s)))).append('\t').append(qc(n.quickCheck(s)))
              .append('\t').append(n.isNormalized(s) ? 1 : 0).append('\t').append(n.spanQuickCheckYes(s))
              .append('\t').append(Long.toHexString(fnv(bounds))).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("norm_blocks.tsv"), blocks.toString(), StandardCharsets.UTF_8);

    // Normalization over the stress strings.
    List<String> stress = stress();
    StringBuilder st = new StringBuilder();
    for (String s : stress) st.append(AnalysisRows.esc(s)).append('\n');
    Files.writeString(out.resolve("norm_strings.txt"), st.toString(), StandardCharsets.UTF_8);
    StringBuilder ns = new StringBuilder();
    for (String form : FORMS) {
      for (String mode : MODES) {
        Normalizer2 n = normalizer(form, mode);
        for (int i = 0; i < stress.size(); i++) {
          String s = stress.get(i);
          int split = s.length() / 2;
          StringBuilder first = new StringBuilder(n.normalize(s.substring(0, split)));
          n.normalizeSecondAndAppend(first, s.substring(split));
          ns.append(form).append('\t').append(mode).append('\t').append(i).append('\t')
              .append(AnalysisRows.esc(n.normalize(s))).append('\t').append(qc(n.quickCheck(s))).append('\t')
              .append(n.spanQuickCheckYes(s)).append('\t').append(AnalysisRows.esc(first.toString())).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("norm_strings.tsv"), ns.toString(), StandardCharsets.UTF_8);

    // UnicodeSet patterns.
    StringBuilder us = new StringBuilder();
    for (String p : unicodeSetPatterns()) us.append(setRow(p)).append('\n');
    Files.writeString(out.resolve("unicode_sets.tsv"), us.toString(), StandardCharsets.UTF_8);

    // Chains.
    List<String> corpus = AnalysisRows.corpus("analysis-icu.txt");
    // Lone surrogates cannot reach a Rust analyzer (its input is a str): the chains run the strings
    // without them.
    List<String> lines = new ArrayList<>(corpus);
    for (String s : stress) {
      boolean lone = false;
      for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i);
        if (Character.isHighSurrogate(c) && (i + 1 == s.length() || !Character.isLowSurrogate(s.charAt(i + 1)))) lone = true;
        if (Character.isLowSurrogate(c) && (i == 0 || !Character.isHighSurrogate(s.charAt(i - 1)))) lone = true;
      }
      if (!lone) lines.add(s);
    }
    Map<String, Supplier<Analyzer>> norm = new LinkedHashMap<>();
    norm.put("n_kw_nfkc_cf", () -> AnalysisRows.chain(KeywordTokenizer::new, ICUNormalizer2Filter::new));
    norm.put("n_ws_nfkc_cf", () -> AnalysisRows.chain(WhitespaceTokenizer::new, ICUNormalizer2Filter::new));
    norm.put("n_kw_folding", () -> AnalysisRows.chain(KeywordTokenizer::new, ICUFoldingFilter::new));
    norm.put("n_ws_folding", () -> AnalysisRows.chain(WhitespaceTokenizer::new, ICUFoldingFilter::new));
    for (String form : FORMS) {
      for (String mode : new String[] {"compose", "decompose"}) {
        norm.put("n_kw_" + form + "_" + mode, () -> {
          try {
            Normalizer2 n = normalizer(form, mode);
            return AnalysisRows.chain(KeywordTokenizer::new, t -> new ICUNormalizer2Filter(t, n));
          } catch (Exception e) {
            throw new RuntimeException(e);
          }
        });
      }
    }
    norm.put("cf_ws_nfkc_cf", () -> AnalysisRows.chain(ICUNormalizer2CharFilter::new, WhitespaceTokenizer::new, t -> t));
    norm.put("cf_kw_nfkc_cf", () -> AnalysisRows.chain(ICUNormalizer2CharFilter::new, KeywordTokenizer::new, t -> t));
    for (int size : new int[] {2, 3, 5}) {
      for (String form : new String[] {"nfkc_cf", "nfc", "utr30"}) {
        for (String mode : new String[] {"compose", "decompose"}) {
          norm.put("cf" + size + "_ws_" + form + "_" + mode, () -> {
            try {
              Normalizer2 n = normalizer(form, mode);
              return AnalysisRows.chain(
                  (Reader r) -> IcuAccess.charFilter(r, n, size), WhitespaceTokenizer::new, t -> t);
            } catch (Exception e) {
              throw new RuntimeException(e);
            }
          });
        }
      }
    }
    AnalysisRows.writeChains(out, norm, lines);

    Map<String, Supplier<Analyzer>> fac = new LinkedHashMap<>();
    String[] specs = {
      "t:whitespace f:icuNormalizer2",
      "t:whitespace f:icuNormalizer2,form=nfc",
      "t:whitespace f:icuNormalizer2,form=nfkc,mode=decompose",
      "t:whitespace f:icuNormalizer2,form=nfkc_cf,filter=[^åäöÅÄÖ]",
      "t:whitespace f:icuNormalizer2,form=nfkc_cf,filter=[:Latin:]",
      "t:whitespace f:icuNormalizer2,filter=[]",
      "t:whitespace f:icuNormalizer2,form=uts46",
      "t:whitespace f:icuFolding",
      "t:whitespace f:icuFolding,filter=[^åäöÅÄÖ]",
      "t:whitespace f:icuFolding,filter=[[:Greek:]_SPACE_[:Cyrillic:]]",
      "c:icuNormalizer2 t:whitespace",
      "c:icuNormalizer2,form=nfc,mode=decompose t:whitespace",
      "c:icuNormalizer2,filter=[^ß] t:whitespace",
      "c:icuNormalizer2,form=nfkc t:keyword",
      "t:icu",
      "t:icu,cjkAsWords=false",
      "t:icu,myanmarAsWords=false",
      "t:icu,cjkAsWords=false,myanmarAsWords=false f:icuFolding",
      "c:icuNormalizer2 t:icu f:icuNormalizer2,form=nfc",
      // refused
      "t:whitespace f:icuNormalizer2,form=bogus",
      "t:whitespace f:icuNormalizer2,mode=Compose",
      "t:whitespace f:icuNormalizer2,filter=[a",
      "t:whitespace f:icuNormalizer2,x=1",
      "t:whitespace f:icuFolding,filter=[z-a]",
      "t:whitespace f:icuFolding,y=2",
      "c:icuNormalizer2,mode=fcd t:whitespace",
      "t:icu,rulefiles=Latn",
      "t:icu,rulefiles=Nope:x.rbbi",
      "t:icu,foo=1",
    };
    for (int i = 0; i < specs.length; i++) fac.put(String.format("c_factory_%02d", i), factory(specs[i]));
    for (Map.Entry<String, Supplier<Analyzer>> e : fac.entrySet()) {
      String rows;
      try (Analyzer a = e.getValue().get()) {
        rows = AnalysisRows.rows(a, corpus);
      } catch (RuntimeException ex) {
        Throwable c = ex.getCause() != null ? ex.getCause() : ex;
        rows = "B\t" + c.getClass().getSimpleName() + "\n";
      }
      Files.writeString(out.resolve(e.getKey() + ".tsv"), rows, StandardCharsets.UTF_8);
    }
    StringBuilder sp = new StringBuilder();
    for (int i = 0; i < specs.length; i++) sp.append(String.format("c_factory_%02d", i)).append('\t').append(specs[i]).append('\n');
    Files.writeString(out.resolve("factory_specs.tsv"), sp.toString(), StandardCharsets.UTF_8);
  }
}
