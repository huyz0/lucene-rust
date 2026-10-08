import com.ibm.icu.text.Collator;
import com.ibm.icu.text.RawCollationKey;
import com.ibm.icu.text.RuleBasedCollator;
import com.ibm.icu.util.ULocale;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.icu.ICUCollationDocValuesField;
import org.apache.lucene.analysis.icu.ICUCollationKeyAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute;
import org.apache.lucene.util.BytesRef;

/**
 * M12 T12.4: ICU collation sort keys (ICU4J 77.1's {@code RuleBasedCollator.getRawCollationKey},
 * what Lucene's {@code ICUCollationKeyAnalyzer}, {@code ICUCollationAttributeFactory} and {@code
 * ICUCollationDocValuesField} index), compared by {@code
 * crates/lucene-analysis-icu/tests/icu_collation_fixtures.rs}.
 *
 * <ul>
 *   <li>{@code coll_strings.txt}: the texts, one per line as space-separated UTF-16 units in hex (lone
 *       surrogates, U+0000, U+FFFE kept): the analysis-icu corpus, curated strings for contractions,
 *       prefixes, expansions, numerics and the identical level, and 1,400 seeded strings from pools
 *       of every kind of mapping the root collation and the tailorings carry.
 *   <li>{@code coll_configs.tsv}: one row per collator -- {@code spec validLocale actualLocale} and
 *       FNV-1a 64 of the concatenated keys of each block of 100 strings, or {@code spec X Exception}
 *       for a refused one. A spec is an ICU locale ID, optionally followed by {@code |name=value}
 *       setter calls applied in order ({@code strength decomposition french caseLevel upper lower
 *       shifted numeric reorder maxVariable}). The specs: every collation bundle of the jar, every
 *       collation type each one lists, IDs that fall back (regions, scripts, three-letter codes,
 *       variants, aliases), the attribute keywords, and setter combinations.
 *   <li>{@code coll_keys_<name>.tsv}: the full key of every string, in hex, for a few collators.
 *   <li>{@code coll_lucene.tsv}: {@code ICUCollationKeyAnalyzer}'s term bytes and {@code
 *       ICUCollationDocValuesField}'s bytes for each corpus line, for three locales.
 * </ul>
 */
public class GenAnalysisIcuCollation {
  static final int BLOCK = 100;

  static final int[][] POOLS = {
    // Latin with the letters tailorings contract or reorder
    {'a', 'b', 'c', 'h', 'l', 'n', 'o', 'u', 'y', 'z', 'A', 'C', 'H', 'L', 'S', 'Z', 'd', 'g', 'j', 's', 't',
      0xe5, 0xe6, 0xf8, 0xe4, 0xf6, 0xfc, 0xf1, 0xe7, 0x151, 0x131, 0x130, 0x10d, 0x17e, 0x161, 0x142, 0x111,
      0xc5, 0xc6, 0xd8, 0xdf, 0x1e9e, 0x149, 0x1c6, 0x1c5, 0x1ea1, 0x1edd},
    // combining marks (discontiguous contractions, FCD segments, Tibetan composites)
    {0x300, 0x301, 0x302, 0x303, 0x308, 0x30a, 0x30c, 0x323, 0x327, 0x328, 0x31b, 0x334, 0x345, 0x35c,
      0x93c, 0x94d, 0x5b0, 0x5bc, 0xf71, 0xf72, 0xf73, 0xf74, 0xf75, 0xf80, 0xf81, 0x302a, 0x3099, 0x309a,
      0x1d165, 0x1d16d, 0x1dce},
    // digits of several scripts, and separators between them
    {'0', '1', '2', '5', '9', '0', '0', 0x660, 0x669, 0x966, 0x967, 0xff10, 0xff19, 0x1d7ce, 0x1040, '.',
      ',', ' ', '-'},
    // variable characters: spaces, punctuation, symbols, currency
    {' ', '\t', '_', '-', ',', ';', '!', '?', '.', '\'', '"', '(', ')', '@', '*', '/', '\\', '&', '#', '%',
      '+', '<', '=', '$', 0xa3, 0x20ac, 0xa5, 0x3000, 0x2010, 0x2014, 0xb7, 0x2026, 0xa9, 0x2122},
    // Hangul syllables and jamo
    {0xac00, 0xac01, 0xb098, 0xd55c, 0xd7a3, 0x1100, 0x1161, 0x11a8, 0x1112, 0x1175, 0x11c2, 0x3131, 0x314f},
    // kana with prolonged sound and iteration marks (prefix contractions), Han
    {0x3042, 0x3044, 0x304b, 0x304c, 0x30a2, 0x30ab, 0x30ac, 0x30fc, 0x309d, 0x309e, 0x30fd, 0x30fe, 0x3005,
      0xff71, 0xff70, 0x4e00, 0x4e8c, 0x5b57, 0x6f22, 0x9fa5, 0x20000, 0x2a6d6, 0x3400, 0xf900, 0x2f00},
    // scripts: Greek, Cyrillic, Arabic, Hebrew, Devanagari, Thai and Lao prevowels, Myanmar, Ethiopic
    {0x3b1, 0x3ac, 0x3c3, 0x3c2, 0x391, 0x430, 0x451, 0x439, 0x419, 0x44c, 0x627, 0x644, 0x623, 0x5d0, 0x5e9,
      0x915, 0x937, 0x93f, 0xe01, 0xe40, 0xe44, 0xe32, 0xe33, 0xe81, 0xec0, 0x1000, 0x1031, 0x1200, 0x10d0},
    // specials: unassigned, private use, noncharacters, lone surrogates, U+0000, U+FFFD, emoji
    {0x378, 0xe000, 0xf8ff, 0xfffe, 0xffff, 0xfffd, 0x0, 0xd800, 0xdc00, 0xdbff, 0xe0080, 0x10ffff, 0x1f600,
      0x1f468, 0x200d, 0x1f3fd, 0xfe0f, 0x1, 0x7f, 0x9f},
  };

  static List<String> strings() throws Exception {
    List<String> out = new ArrayList<>(AnalysisRows.corpus("analysis-icu.txt"));
    String[] curated = {
      "", "a", "A", "ab", "aB", "Ab", "AB", "ch", "Ch", "CH", "cH", "c", "h", "chch", "cha", "lla", "ll", "LL",
      "ñ", "ñ", "Ñ", "aa", "Aa", "AA", "å", "å", "ae", "æ", "oe", "ö", "ö", "ő",
      "dz", "dž", "dzs", "ly", "ny", "gy", "cs", "zs", "rr", "ı", "i", "I", "İ", "i̇", "ß", "ss", "SS",
      "ẞ", "ŉ", "ǆ", "ǅ", "Ǆ", "ä", "ä", "ạ̈", "ạ̈", "ạ̈", "ạ́",
      "ạ́", "เก", "กเ", "ເກ", "カー", "かー",
      "カゝ", "カヽ", "ガヾ", "ཱི", "ཱི", "ཱུ", "ཱུ",
      "ཱྀ", "ཀཱི", "1", "2", "10", "01", "001", "1000000", "2000000", "99999999", "1.5", "1,5",
      "0", "00", "0000000000", "12345678901234567890", "1".repeat(260), "0".repeat(10) + "7",
      "١٢", "१२", "１２", "𝟏", "a1b", "a10b", "a2b", "가",
      "각", "가", "각", "한글", "一", "二", "字",
      "𠀀", "㐀", "豈", "⼀", "￾", "￿", "�", "\u0000", "a\u0000b",
      "\ud800", "\udc00", "a\ud800", "\ud800a", "\udc00\ud800", "͸", "", "󠂀",
      "􏿿", " ", "  ", "a b", "a-b", "a_b", "ab ", "-", "--", "$", "€", "£", "@", "*", "a­",
      "co-op", "coop", "co op", "résumé", "resume", "Résumé", "côte", "coté", "côté", "cote", "peach", "péché",
      "pêche", "pèché", "Łódź", "ž", "ž", "š", "ć", "č", "đ",
      "ǆ", "ı", "ğ", "ş", "ç", "ă", "ș", "ț", "ð", "þ",
      "ɔ", "ə", "ŋ", "ħ", "ά", "ά", "ά", "й", "й",
      "ё", "ё", "آ", "آ", "क़", "क़", "क्ष",
      "க்ஷ", "ကျ", "ေက", "េក", "ẛ̣",
      "ὠ0", "😀", "👨‍👩", "Ạ̊", "Å",
      "Ω", "ḍ̇", "ḍ̇", "q̣̇", "Ą̊", "̀ͅ",
      "̀ͅ", "ุ่", "ุ่", "゙゚", "ᴖ5ᴖd",
      "𝅥𝅭", "𝅥𝅭",
    };
    Collections.addAll(out, curated);
    Random r = new Random(77);
    for (int i = 0; i < 1400; i++) {
      StringBuilder b = new StringBuilder();
      int n = 1 + r.nextInt(i % 70 == 69 ? 120 : 10);
      int pool = r.nextInt(POOLS.length);
      for (int k = 0; k < n; k++) {
        if (r.nextInt(4) == 0) pool = r.nextInt(POOLS.length);
        int[] p = POOLS[pool];
        b.appendCodePoint(p[r.nextInt(p.length)]);
      }
      out.add(b.toString());
    }
    return out;
  }

  static String units(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      if (i > 0) b.append(' ');
      b.append(String.format("%04x", (int) s.charAt(i)));
    }
    return b.toString();
  }

  static String hex(byte[] bytes, int length) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < length; i++) b.append(String.format("%02x", bytes[i] & 0xff));
    return b.toString();
  }

  static String exceptionName(Throwable e) {
    return e.getClass().getSimpleName();
  }

  /** A collator from a spec: an ICU locale ID, then {@code |name=value} setter calls. */
  static RuleBasedCollator collator(String spec) throws Exception {
    String[] parts = spec.split("\\|");
    RuleBasedCollator c = (RuleBasedCollator) Collator.getInstance(new ULocale(parts[0]));
    for (int i = 1; i < parts.length; i++) {
      String[] kv = parts[i].split("=", 2);
      String v = kv[1];
      switch (kv[0]) {
        case "strength" -> c.setStrength(Integer.parseInt(v));
        case "decomposition" -> c.setDecomposition(Integer.parseInt(v));
        case "french" -> c.setFrenchCollation(v.equals("1"));
        case "caseLevel" -> c.setCaseLevel(v.equals("1"));
        case "upper" -> c.setUpperCaseFirst(v.equals("1"));
        case "lower" -> c.setLowerCaseFirst(v.equals("1"));
        case "shifted" -> c.setAlternateHandlingShifted(v.equals("1"));
        case "numeric" -> c.setNumericCollation(v.equals("1"));
        case "maxVariable" -> c.setMaxVariable(Integer.parseInt(v));
        case "reorder" -> {
          String[] codes = v.isEmpty() ? new String[0] : v.split(",");
          int[] order = new int[codes.length];
          for (int k = 0; k < codes.length; k++) order[k] = Integer.parseInt(codes[k]);
          c.setReorderCodes(order);
        }
        default -> throw new IllegalStateException(kv[0]);
      }
    }
    return c;
  }

  static long fnv(long h, byte[] bytes, int length) {
    for (int i = 0; i < length; i++) {
      h = (h ^ (bytes[i] & 0xff)) * 0x100000001b3L;
    }
    return h;
  }

  static List<String> specs() throws Exception {
    List<String> specs = new ArrayList<>();
    // Every collation bundle of the jar, and every type each one lists.
    String jar = Collator.class.getProtectionDomain().getCodeSource().getLocation().getPath();
    TreeSet<String> ids = new TreeSet<>();
    try (JarFile j = new JarFile(jar)) {
      for (JarEntry e : Collections.list(j.entries())) {
        String n = e.getName();
        String dir = "com/ibm/icu/impl/data/icudata/coll/";
        if (n.startsWith(dir) && n.endsWith(".res") && !n.substring(dir.length()).contains("/")) {
          String id = n.substring(dir.length(), n.length() - 4);
          if (!id.equals("res_index")) ids.add(id);
        }
      }
    }
    for (String id : ids) {
      specs.add(id);
      TreeSet<String> types = new TreeSet<>();
      for (String t : Collator.getKeywordValuesForLocale("collation", new ULocale(id), false)) types.add(t);
      for (String t : types) {
        if (!t.equals("standard")) specs.add(id + "@collation=" + t);
      }
    }
    // IDs that fall back, alias, canonicalize.
    Collections.addAll(specs,
        "", "root", "ROOT", "und", "und_DE", "xx", "xx_YY", "de_CH", "de_DE", "de_AT", "de__PHONEBOOK",
        "de_DE_PHONEBOOK", "en_GB", "en_US", "en_US_POSIX", "fr_CH", "fr_CA", "es_MX", "es__TRADITIONAL",
        "pt_PT", "pt_BR", "zh_HK", "zh_MO", "zh_SG", "zh_Hans_HK", "zh_Hant", "zh_TW", "zh_CN", "zh__PINYIN",
        "zh@collation=stroke", "zh_Hant@collation=pinyin", "yue_HK", "yue_Hans", "yue_CN", "sr_Latn_ME",
        "sr_Latn_BA", "sr_ME", "sr_Cyrl_XK", "sr_XK", "bs_Cyrl_BA", "pa_PK", "pa_Arab", "pa_Guru_IN", "uz_Cyrl",
        "ff_Adlm_GN", "ff_SN", "deu", "deu_DEU", "fra_FRA", "DE-de", "de-AT", "iw", "he", "in", "id", "no",
        "nb", "nn", "mo", "ro_MD", "sh", "sh_YU", "tl", "fil", "ji", "yi", "ku_TR", "az_Cyrl", "ja_JP",
        "ja@collation=unihan", "ko@collation=search", "ko@collation=searchjl", "de@collation=search",
        "de@collation=searchxyz", "de@collation=nonsense", "de@collation=default", "en@collation=PhoneBook",
        "de@collation=PHONEBOOK", "th_TH", "th@collation=standard", "sv@collation=traditional",
        "sv@collation=reformed", "da_DK", "nb_NO", "el_GR", "ar_SA", "ar_EG", "fa_AF", "hi_IN", "ta_LK");
    // Attribute keywords.
    String[] kw = {
      "colStrength=primary", "colStrength=secondary", "colStrength=tertiary", "colStrength=quaternary",
      "colStrength=identical", "colStrength=IDENTICAL", "colStrength=bogus", "colAlternate=shifted",
      "colAlternate=non-ignorable", "colAlternate=x", "colBackwards=yes", "colBackwards=no",
      "colBackwards=maybe", "colCaseLevel=yes", "colCaseFirst=upper", "colCaseFirst=lower",
      "colCaseFirst=no", "colCaseFirst=x", "colNormalization=yes", "colNormalization=no", "colNumeric=yes",
      "colNumeric=no", "colReorder=Grek-Latn", "colReorder=digit-space-Latn", "colReorder=Zzzz-Latn",
      "colReorder=Latn-Zzzz-Grek", "colReorder=Latn-Latn", "colReorder=Hani", "colReorder=Hira-Kana",
      "colReorder=Xxxx", "colReorder=others", "colReorder=Cyrl-punct-currency-symbol",
      "kv=space", "kv=punct", "kv=symbol", "kv=currency", "kv=digit", "colHiraganaQuaternary=yes",
      "variableTop=0020", "colStrength=quaternary;colAlternate=shifted",
      "colAlternate=shifted;kv=symbol", "colCaseFirst=upper;colCaseLevel=yes;colStrength=primary",
      "colNumeric=yes;colStrength=identical", "collation=phonebook;colStrength=primary",
    };
    for (String base : new String[] {"en", "de", "ja", "da"}) {
      for (String k : kw) specs.add(base + "@" + k);
    }
    // Setter combinations.
    String[] bases = {"", "en", "fr", "fr_CA", "da", "ja", "zh", "ko", "th", "ar", "de@collation=phonebook",
      "es@collation=traditional", "cs", "sk", "hu", "lt", "tr", "vi", "ln@collation=phonetic"};
    String[] setters = {
      "|strength=0", "|strength=1", "|strength=3", "|strength=15", "|strength=7", "|decomposition=17",
      "|decomposition=16", "|decomposition=3", "|french=1", "|french=0", "|caseLevel=1",
      "|caseLevel=1|strength=0", "|caseLevel=1|upper=1", "|upper=1", "|lower=1", "|upper=1|lower=1",
      "|upper=1|strength=3", "|lower=1|caseLevel=1|strength=1", "|shifted=1", "|shifted=1|strength=3",
      "|shifted=1|strength=15", "|shifted=1|maxVariable=4096", "|shifted=1|maxVariable=4098",
      "|shifted=1|maxVariable=4099", "|shifted=1|maxVariable=-1", "|maxVariable=4100", "|numeric=1",
      "|numeric=1|strength=0", "|reorder=20,25", "|reorder=4100,4096", "|reorder=-1", "|reorder=103",
      "|reorder=", "|reorder=25,25", "|reorder=17,22,20|strength=3", "|reorder=4097,4098,4099,4100,25",
      "|reorder=103,25", "|reorder=25,103,14", "|reorder=-1,25", "|reorder=12,25|reorder=-1",
      "|french=1|strength=15|shifted=1|caseLevel=1|upper=1|numeric=1|decomposition=17",
    };
    for (String base : bases) {
      for (String s : setters) specs.add(base + s);
    }
    return specs;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_icu_collation");
    Files.createDirectories(out);
    List<String> strings = strings();
    StringBuilder sb = new StringBuilder();
    for (String s : strings) sb.append(units(s)).append('\n');
    Files.writeString(out.resolve("coll_strings.txt"), sb.toString(), StandardCharsets.UTF_8);

    StringBuilder rows = new StringBuilder();
    RawCollationKey key = new RawCollationKey();
    for (String spec : specs()) {
      RuleBasedCollator c;
      try {
        c = collator(spec);
      } catch (Exception e) {
        rows.append(spec).append("\tX\t").append(exceptionName(e)).append('\n');
        continue;
      }
      rows.append(spec)
          .append('\t').append(c.getLocale(ULocale.VALID_LOCALE).getName())
          .append('\t').append(c.getLocale(ULocale.ACTUAL_LOCALE).getName());
      long h = 0xcbf29ce484222325L;
      for (int i = 0; i < strings.size(); i++) {
        c.getRawCollationKey(strings.get(i), key);
        h = fnv(h, key.bytes, key.size);
        if (i % BLOCK == BLOCK - 1 || i == strings.size() - 1) {
          rows.append('\t').append(Long.toHexString(h));
          h = 0xcbf29ce484222325L;
        }
      }
      rows.append('\n');
    }
    Files.writeString(out.resolve("coll_configs.tsv"), rows.toString(), StandardCharsets.UTF_8);

    String[][] full = {
      {"root", ""}, {"de_phonebook", "de@collation=phonebook"}, {"ja_identical", "ja|strength=15"},
      {"en_shifted_quaternary", "en|shifted=1|strength=3"}, {"fr_ca", "fr_CA"},
      {"th_numeric_caselevel", "th|numeric=1|caseLevel=1|upper=1"}, {"da_fcd", "da|decomposition=17"},
    };
    for (String[] f : full) {
      RuleBasedCollator c = collator(f[1]);
      StringBuilder k = new StringBuilder();
      for (String s : strings) {
        c.getRawCollationKey(s, key);
        k.append(hex(key.bytes, key.size)).append('\n');
      }
      Files.writeString(out.resolve("coll_keys_" + f[0] + ".tsv"), k.toString(), StandardCharsets.UTF_8);
    }

    // Lucene's classes over the corpus.
    StringBuilder lucene = new StringBuilder();
    for (String loc : new String[] {"", "de@collation=phonebook", "ja"}) {
      Collator c = Collator.getInstance(new ULocale(loc));
      Analyzer a = new ICUCollationKeyAnalyzer(c);
      ICUCollationDocValuesField field = new ICUCollationDocValuesField("f", c);
      for (String line : AnalysisRows.corpus("analysis-icu.txt")) {
        try (TokenStream ts = a.tokenStream("f", line)) {
          TermToBytesRefAttribute term = ts.addAttribute(TermToBytesRefAttribute.class);
          ts.reset();
          StringBuilder terms = new StringBuilder();
          while (ts.incrementToken()) {
            BytesRef b = term.getBytesRef();
            if (terms.length() > 0) terms.append(',');
            terms.append(hex(java.util.Arrays.copyOfRange(b.bytes, b.offset, b.offset + b.length), b.length));
          }
          ts.end();
          field.setStringValue(line);
          BytesRef dv = field.binaryValue();
          lucene.append(loc).append('\t').append(terms).append('\t')
              .append(hex(java.util.Arrays.copyOfRange(dv.bytes, dv.offset, dv.offset + dv.length), dv.length))
              .append('\n');
        }
      }
      a.close();
    }
    Files.writeString(out.resolve("coll_lucene.tsv"), lucene.toString(), StandardCharsets.UTF_8);
  }

  private GenAnalysisIcuCollation() {}
}
