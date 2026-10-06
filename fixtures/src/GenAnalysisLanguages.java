import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.ar.*;
import org.apache.lucene.analysis.bg.*;
import org.apache.lucene.analysis.bn.*;
import org.apache.lucene.analysis.br.*;
import org.apache.lucene.analysis.ca.CatalanAnalyzer;
import org.apache.lucene.analysis.ckb.*;
import org.apache.lucene.analysis.core.KeywordTokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.cz.*;
import org.apache.lucene.analysis.da.DanishAnalyzer;
import org.apache.lucene.analysis.de.*;
import org.apache.lucene.analysis.el.*;
import org.apache.lucene.analysis.es.*;
import org.apache.lucene.analysis.et.EstonianAnalyzer;
import org.apache.lucene.analysis.eu.BasqueAnalyzer;
import org.apache.lucene.analysis.fa.*;
import org.apache.lucene.analysis.fi.*;
import org.apache.lucene.analysis.fr.*;
import org.apache.lucene.analysis.ga.*;
import org.apache.lucene.analysis.gl.*;
import org.apache.lucene.analysis.hi.*;
import org.apache.lucene.analysis.hu.*;
import org.apache.lucene.analysis.hy.ArmenianAnalyzer;
import org.apache.lucene.analysis.id.*;
import org.apache.lucene.analysis.in.IndicNormalizationFilter;
import org.apache.lucene.analysis.it.*;
import org.apache.lucene.analysis.lt.LithuanianAnalyzer;
import org.apache.lucene.analysis.lv.*;
import org.apache.lucene.analysis.ne.NepaliAnalyzer;
import org.apache.lucene.analysis.nl.DutchAnalyzer;
import org.apache.lucene.analysis.no.*;
import org.apache.lucene.analysis.pt.*;
import org.apache.lucene.analysis.ro.*;
import org.apache.lucene.analysis.ru.*;
import org.apache.lucene.analysis.sr.*;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.sv.*;
import org.apache.lucene.analysis.ta.TamilAnalyzer;
import org.apache.lucene.analysis.te.*;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tr.*;

/**
 * M11 T11.6: the per-language packages. Every language analyzer (default stop set, and some with a
 * stem exclusion set), and the stemming and normalization filters on their own, over {@code
 * fixtures/corpus/analysis-lang.txt} (sentences written for this project) as {@code <chain>.tsv}
 * ({@link AnalysisRows}' rows), {@code normalize.words} ({@code Analyzer.normalize} of each), and {@code lang.words}: each stemmer and normalizer over words
 * built from its own suffixes, prefixes and characters (harvested from Lucene's sources into
 * {@link #AFFIXES}, RSLP's exception words as they are) on short bases, as "filter\tword\tresult". {@code
 * crates/lucene-analysis/tests/analysis_lang_fixtures.rs} compares.
 */
public class GenAnalysisLanguages {

  static CharArraySet set(String... w) {
    return AnalysisRows.set(false, w);
  }

  /** The per-term filters, each over a {@link KeywordTokenizer} for {@code lang.words}. */
  static Map<String, Function<TokenStream, TokenStream>> filters() {
    Map<String, Function<TokenStream, TokenStream>> f = new LinkedHashMap<>();
    f.put("GermanLight", GermanLightStemFilter::new);
    f.put("GermanMinimal", GermanMinimalStemFilter::new);
    f.put("German", GermanStemFilter::new);
    f.put("GermanNormalization", GermanNormalizationFilter::new);
    f.put("FrenchLight", FrenchLightStemFilter::new);
    f.put("FrenchMinimal", FrenchMinimalStemFilter::new);
    f.put("SpanishLight", SpanishLightStemFilter::new);
    f.put("SpanishMinimal", SpanishMinimalStemFilter::new);
    f.put("SpanishPlural", SpanishPluralStemFilter::new);
    f.put("ItalianLight", ItalianLightStemFilter::new);
    f.put("PortugueseLight", PortugueseLightStemFilter::new);
    f.put("PortugueseMinimal", PortugueseMinimalStemFilter::new);
    f.put("Portuguese", PortugueseStemFilter::new);
    f.put("Galician", GalicianStemFilter::new);
    f.put("GalicianMinimal", GalicianMinimalStemFilter::new);
    f.put("SwedishLight", SwedishLightStemFilter::new);
    f.put("SwedishMinimal", SwedishMinimalStemFilter::new);
    f.put("NorwegianLight", NorwegianLightStemFilter::new);
    f.put("NorwegianLightNynorsk", t -> new NorwegianLightStemFilter(t, 2));
    f.put("NorwegianLightBoth", t -> new NorwegianLightStemFilter(t, 3));
    f.put("NorwegianMinimal", NorwegianMinimalStemFilter::new);
    f.put("NorwegianMinimalNynorsk", t -> new NorwegianMinimalStemFilter(t, 2));
    f.put("NorwegianNormalization", NorwegianNormalizationFilter::new);
    f.put("FinnishLight", FinnishLightStemFilter::new);
    f.put("HungarianLight", HungarianLightStemFilter::new);
    f.put("RussianLight", RussianLightStemFilter::new);
    f.put("Bulgarian", BulgarianStemFilter::new);
    f.put("Czech", CzechStemFilter::new);
    f.put("Latvian", LatvianStemFilter::new);
    f.put("Indonesian", IndonesianStemFilter::new);
    f.put("IndonesianInflectional", t -> new IndonesianStemFilter(t, false));
    f.put("Hindi", HindiStemFilter::new);
    f.put("HindiNormalization", HindiNormalizationFilter::new);
    f.put("Bengali", BengaliStemFilter::new);
    f.put("BengaliNormalization", BengaliNormalizationFilter::new);
    f.put("Telugu", TeluguStemFilter::new);
    f.put("TeluguNormalization", TeluguNormalizationFilter::new);
    f.put("Sorani", SoraniStemFilter::new);
    f.put("SoraniNormalization", SoraniNormalizationFilter::new);
    f.put("Arabic", ArabicStemFilter::new);
    f.put("ArabicNormalization", ArabicNormalizationFilter::new);
    f.put("Persian", PersianStemFilter::new);
    f.put("PersianNormalization", PersianNormalizationFilter::new);
    f.put("IndicNormalization", IndicNormalizationFilter::new);
    f.put("RomanianNormalization", RomanianNormalizationFilter::new);
    f.put("IrishLowerCase", IrishLowerCaseFilter::new);
    f.put("TurkishLowerCase", TurkishLowerCaseFilter::new);
    f.put("Apostrophe", ApostropheFilter::new);
    f.put("Greek", GreekStemFilter::new);
    f.put("GreekLowerCase", GreekLowerCaseFilter::new);
    f.put("Brazilian", BrazilianStemFilter::new);
    f.put("SerbianNormalization", SerbianNormalizationFilter::new);
    f.put("SerbianNormalizationRegular", SerbianNormalizationRegularFilter::new);
    return f;
  }

  /** Which harvested affixes feed a filter (by default its own name). */
  static String affixKey(String filter) {
    return switch (filter) {
      case "NorwegianLightNynorsk", "NorwegianLightBoth" -> "NorwegianLight";
      case "NorwegianMinimalNynorsk" -> "NorwegianMinimal";
      case "IndonesianInflectional" -> "Indonesian";
      case "PortugueseMinimal" -> "Portuguese";
      case "GalicianMinimal" -> "Galician";
      case "NorwegianNormalization" -> "GermanNormalization";
      default -> filter;
    };
  }

  /** Hand-written inputs for the context rules the harvested affixes do not reach. */
  static final Map<String, String[]> EXTRA = new LinkedHashMap<>();

  static {
    String k = "\u0995", h = "\u09CD";
    EXTRA.put("BengaliNormalization", new String[] {
      k + h + "\u09BF", k + k + h + "\u09BF", k + h + "\u09AF\u09BE", k + h + "\u09AF", k + k + h + "\u09AF",
      k + h + "\u09AC", k + k + k + h + k + h + "\u09AC", k + k + h + "\u09AC", "\u09AC" + k, k + "\u09AC",
      k + "\u0983", k + k + k + "\u0983", k + "\u0983" + k, "\u0981" + k + "\u0981", k + "\u09C0\u09C2\u0999",
      "\u09B6\u09B7\u09A3\u09DC\u09DD\u09CE", k + k + "\u09AF", "\u09AF", h + "\u09AF"
    });
    EXTRA.put("TeluguNormalization", new String[] {
      "\u0C15\u0C46\u0C56", "\u0C12\u0C55", "\u0C12\u0C4C", "\u0C12", "\u0C46", "\u0C15\u0C03\u200D\u200C",
      "\u0C00\u0C01\u0C14\u0C10\u0C06\u0C08\u0C0A\u0C40\u0C42\u0C47\u0C4B"
    });
    EXTRA.put("TurkishLowerCase", new String[] {
      "I\u0307", "I\u0300\u0307x", "AIB", "\u0130", "I\u0300", "\uD801\uDC00", "\u0131I", "I", "II\u0307", "I\u0300a"
    });
    EXTRA.put("HindiNormalization", new String[] {"\u0928\u094D", "\u0928\u094D\u0915", "\u0928", "\u0915\u093C\u200D\u200C"});
    EXTRA.put("SoraniNormalization", new String[] {
      "\u0647\u200C\u0628", "\u200C", "\u0628\u0647", "\u0631\u0628", "\u0628\u0631", "\u0628\u200E\u0628"
    });
    EXTRA.put("Bulgarian", new String[] {"абвгдежи", "кукатата", "абвгдъл", "абвгдъни", "работата", "абвгдета", "абвгдътата"});
    EXTRA.put("GermanNormalization", new String[] {"aue", "uue", "oeu", "queue", "Maße", "ß", "äöüß", "yae", "iue"});
    EXTRA.put("German", new String[] {
      "Häuser", "abschließen", "a1", "Schlüssel", "Kinderinnen", "erinnern", "geigegegen", "Sitzplätze", "Bahnhöfe",
      "Bücher", "trinkend", "gestern", "ΣΟΦΙΑΣ", "İstanbul", "Straßenbahnen", "Gegebenheiten", "Schachspiel", "ii"
    });
  }

  /** Short bases in the filter's script, by length 1, 2, 3, 6. */
  static String[] bases(String filter) {
    if (filter.startsWith("Greek")) return new String[] {"κ", "κα", "καλ", "καλοκα"};
    if (filter.startsWith("Russian") || filter.startsWith("Bulgarian") || filter.startsWith("Serbian")) return new String[] {"д", "ко", "дом", "красив"};
    if (filter.startsWith("Hindi") || filter.startsWith("Indic")) return new String[] {"क", "कम", "कमल", "कमलनयन"};
    if (filter.startsWith("Bengali")) return new String[] {"ক", "কম", "কমল", "কমলনয়ন"};
    if (filter.startsWith("Telugu")) return new String[] {"క", "కల", "కలమ", "కలమునక"};
    if (filter.startsWith("Sorani")) return new String[] {"ک", "کت", "کتێ", "کتێبخان"};
    if (filter.startsWith("Arabic") || filter.startsWith("Persian")) return new String[] {"ك", "كت", "كتب", "مكتبتن"};
    return new String[] {"k", "ka", "bor", "tankar"};
  }

  /** Every code-point decomposition {@code IndicNormalizer} composes, in each of its scripts. */
  static List<String> indicWords() {
    List<String> w = new ArrayList<>();
    int[] bases = {0x900, 0x980, 0xA00, 0xA80, 0xB00, 0xB80, 0xC00, 0xC80, 0xD00};
    int[][] seqs = {
      {0x05, 0x3E, 0x45}, {0x05, 0x3E, 0x46}, {0x05, 0x3E, 0x47}, {0x05, 0x3E, 0x48}, {0x05, 0x3E}, {0x05, 0x45},
      {0x05, 0x46}, {0x05, 0x47}, {0x05, 0x48}, {0x05, 0x49}, {0x05, 0x4A}, {0x05, 0x4B}, {0x05, 0x4C},
      {0x06, 0x45}, {0x06, 0x46}, {0x06, 0x47}, {0x06, 0x48}, {0x07, 0x57}, {0x09, 0x41}, {0x09, 0x57},
      {0x0E, 0x46}, {0x0F, 0x45}, {0x0F, 0x46}, {0x0F, 0x47}, {0x0F, 0x57}, {0x12, 0x3E}, {0x12, 0x4C},
      {0x12, 0x55}, {0x12, 0x57}, {0x13, 0x57}, {0x14, 0x57}, {0x3E, 0x45}, {0x3E, 0x46}, {0x3E, 0x47},
      {0x3E, 0x48}, {0x46, 0x3E}, {0x46, 0x42, 0x55}, {0x46, 0x56}, {0x46, 0x57}, {0x47, 0x3E}, {0x47, 0x57},
      {0x4A, 0x55}, {0x05, 0x4D, 0xFF}, {0x09, 0x4D, 0xFF}, {0x0A, 0x4D, 0xFF}, {0x13, 0x4D, 0xFF},
      {0x15, 0x4D, 0xFF}, {0x72, 0x3F}, {0x72, 0x40}, {0x72, 0x47}, {0x73, 0x41}, {0x73, 0x42}, {0x73, 0x4B},
      {0x05, 0x3E, 0x200D}, {0x05}, {0x05, 0x41}
    };
    for (int base : bases) {
      for (int[] s : seqs) {
        StringBuilder b = new StringBuilder("क");
        for (int c : s) b.append((char) (c == 0xFF ? 0x200D : c >= 0x200D ? c : base + c));
        w.add(b.toString());
        b.append('x');
        w.add(b.toString());
        w.add(b.substring(1, b.length() - 1));
      }
    }
    return w;
  }

  static String apply(Function<TokenStream, TokenStream> f, String word) throws Exception {
    Tokenizer t = new KeywordTokenizer();
    t.setReader(new StringReader(word));
    try (TokenStream s = f.apply(t)) {
      CharTermAttribute a = s.addAttribute(CharTermAttribute.class);
      s.reset();
      String r = s.incrementToken() ? a.toString() : "<none>";
      s.end();
      return r;
    }
  }

  static void writeWords(Path out) throws Exception {
    StringBuilder b = new StringBuilder();
    for (Map.Entry<String, Function<TokenStream, TokenStream>> e : filters().entrySet()) {
      Random rnd = new Random(e.getKey().hashCode());
      Set<String> words = new LinkedHashSet<>();
      String[] affixes = AFFIXES.getOrDefault(affixKey(e.getKey()), new String[0]);
      String[] bases = bases(e.getKey());
      for (String a : affixes) {
        words.add(a);
        words.add(bases[1] + a);
        words.add(bases[3] + a);
        words.add(a + bases[3]);
        words.add(bases[2] + a + affixes[rnd.nextInt(affixes.length)]);
        words.add((bases[0] + a).toUpperCase(java.util.Locale.ROOT));
      }
      // RSLP's exception words, as they are.
      for (String w : AFFIXES.getOrDefault(affixKey(e.getKey()) + "Exceptions", new String[0])) words.add(w);
      if (e.getKey().equals("IndicNormalization")) words.addAll(indicWords());
      for (String w : EXTRA.getOrDefault(e.getKey(), new String[0])) words.add(w);
      if (e.getKey().equals("Latvian")) {
        // Every palatalized pair before every palatalizing suffix.
        for (String pair : new String[] {"kš", "ņņ", "pj", "bj", "mj", "vj", "šņ", "žņ", "šļ", "žļ", "ļņ", "ļļ", "č", "ļ", "ņ", "t"}) {
          for (String suffix : new String[] {"u", "a", "i", "us", "os", "iem", "s"}) words.add("ka" + pair + suffix);
        }
      }
      if (e.getKey().equals("IrishLowerCase")) {
        for (String w : new String[] {"nAthair", "tÉin", "nÁit", "tUbh", "nOileán", "n", "nX", "tIr", "BAILE", "na", "t"}) words.add(w);
      }
      if (e.getKey().equals("Apostrophe")) {
        for (String w : new String[] {"Türkiye'de", "Ankara’nın", "'baş", "son'", "yok"}) words.add(w);
      }
      for (String w : words) {
        b.append(e.getKey()).append('\t').append(AnalysisRows.esc(w)).append('\t')
            .append(AnalysisRows.esc(apply(e.getValue(), w))).append('\n');
      }
    }
    Files.writeString(out.resolve("lang.words"), b.toString(), StandardCharsets.UTF_8);
  }

  static Map<String, Supplier<Analyzer>> chains() {
    Map<String, Supplier<Analyzer>> c = new LinkedHashMap<>();
    c.put("arabic", ArabicAnalyzer::new);
    c.put("arabic_excl", () -> new ArabicAnalyzer(ArabicAnalyzer.getDefaultStopSet(), set("الأطفال")));
    c.put("armenian", ArmenianAnalyzer::new);
    c.put("basque", BasqueAnalyzer::new);
    c.put("bengali", BengaliAnalyzer::new);
    c.put("brazilian", BrazilianAnalyzer::new);
    c.put("bulgarian", BulgarianAnalyzer::new);
    c.put("catalan", CatalanAnalyzer::new);
    c.put("czech", CzechAnalyzer::new);
    c.put("danish", DanishAnalyzer::new);
    c.put("dutch", DutchAnalyzer::new);
    c.put("dutch_no_dict", () -> new DutchAnalyzer(DutchAnalyzer.getDefaultStopSet(), set("kinderen"), CharArrayMapHolder.EMPTY));
    c.put("estonian", EstonianAnalyzer::new);
    c.put("finnish", FinnishAnalyzer::new);
    c.put("french", FrenchAnalyzer::new);
    c.put("french_excl", () -> new FrenchAnalyzer(FrenchAnalyzer.getDefaultStopSet(), set("enfants", "châteaux")));
    c.put("galician", GalicianAnalyzer::new);
    c.put("german", GermanAnalyzer::new);
    c.put("german_excl", () -> new GermanAnalyzer(GermanAnalyzer.getDefaultStopSet(), set("häuser")));
    c.put("greek", GreekAnalyzer::new);
    c.put("hindi", HindiAnalyzer::new);
    c.put("hungarian", HungarianAnalyzer::new);
    c.put("indonesian", IndonesianAnalyzer::new);
    c.put("irish", IrishAnalyzer::new);
    c.put("italian", ItalianAnalyzer::new);
    c.put("latvian", LatvianAnalyzer::new);
    c.put("lithuanian", LithuanianAnalyzer::new);
    c.put("nepali", NepaliAnalyzer::new);
    c.put("norwegian", NorwegianAnalyzer::new);
    c.put("persian", PersianAnalyzer::new);
    c.put("portuguese", PortugueseAnalyzer::new);
    c.put("romanian", RomanianAnalyzer::new);
    c.put("russian", RussianAnalyzer::new);
    c.put("serbian", SerbianAnalyzer::new);
    c.put("sorani", SoraniAnalyzer::new);
    c.put("spanish", SpanishAnalyzer::new);
    c.put("swedish", SwedishAnalyzer::new);
    c.put("swedish_excl", () -> new SwedishAnalyzer(SwedishAnalyzer.getDefaultStopSet(), set("barnen")));
    c.put("tamil", TamilAnalyzer::new);
    c.put("telugu", TeluguAnalyzer::new);
    c.put("turkish", TurkishAnalyzer::new);
    // The filters' own behaviour is lang.words'; three run in a stream for their offsets and types.
    c.put("ws_turkish_lowercase_apostrophe", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new TurkishLowerCaseFilter(new ApostropheFilter(t))));
    c.put("ws_irish_lowercase", () -> AnalysisRows.chain(WhitespaceTokenizer::new, IrishLowerCaseFilter::new));
    c.put("ws_german_normalization_minimal", () -> AnalysisRows.chain(WhitespaceTokenizer::new, t -> new GermanMinimalStemFilter(new GermanNormalizationFilter(t))));
    c.put("persian_char_filter", () -> AnalysisRows.chain(PersianCharFilter::new, WhitespaceTokenizer::new, t -> t));
    return c;
  }

  /** An empty {@code CharArrayMap<String>} (no stem overrides). */
  static final class CharArrayMapHolder {
    static final org.apache.lucene.analysis.CharArrayMap<String> EMPTY = new org.apache.lucene.analysis.CharArrayMap<>(0, false);
  }

  static final String[] NORMALIZE = {
    "ÀÉÎ Straße", "İSTANBUL'DA", "ΣΟΦΙΑ Άλφα", "كِتَاب ١٢٣", "کتاب‌ها", "हिन्दी ३", "ক্ষমা", "తెలుగు", "Ünïcödé", "l'Église", "tAthair", "ŞŢ"
  };

  /** {@code Analyzer.normalize} of every analyzer chain: "chain\ttext\thex". */
  static void writeNormalize(Path out) throws Exception {
    StringBuilder b = new StringBuilder();
    for (Map.Entry<String, Supplier<Analyzer>> e : chains().entrySet()) {
      try (Analyzer a = e.getValue().get()) {
        for (String t : NORMALIZE) {
          b.append(e.getKey()).append('\t').append(AnalysisRows.esc(t)).append('\t')
              .append(AnalysisRows.hex(a.normalize("f", t))).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("normalize.words"), b.toString(), StandardCharsets.UTF_8);
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_lang");
    Files.createDirectories(out);
    List<String> lines = AnalysisRows.corpus("analysis-lang.txt");
    AnalysisRows.writeChains(out, chains(), lines);
    writeWords(out);
    writeNormalize(out);
  }

  // ---- Lucene's own suffixes, prefixes and characters, per stemmer (harvested from the 10.5.0
  // sources and RSLP rule files by a one-off script; see the class javadoc).

  static final Map<String, String[]> AFFIXES = new LinkedHashMap<>();

  static {
    AFFIXES.put("GermanLight", new String[] {
      "\u00e4", "\u00e0", "\u00e1", "\u00e2", "a", "\u00f6", "\u00f2", "\u00f3",
      "\u00f4", "o", "\u00ef", "\u00ec", "\u00ed", "\u00ee", "i", "\u00fc",
      "\u00f9", "\u00fa", "\u00fb", "u", "b", "d", "f", "g",
      "h", "k", "l", "m", "n", "t", "e", "r",
      "s",
    });
    AFFIXES.put("GermanMinimal", new String[] {
      "\u00e4", "a", "\u00f6", "o", "\u00fc", "u", "n", "e",
      "s", "r",
    });
    AFFIXES.put("German", new String[] {
      "de-DE", "nd", "em", "er", "erin*", "gege", "e", "s",
      "n", "t", "z", "x", "*", "\u00e4", "a", "\u00f6",
      "o", "\u00fc", "u", "\u00df", "c", "h", "$", "\u00a7",
      "i", "%", "&", "g", "#", "!",
    });
    AFFIXES.put("GermanNormalization", new String[] {
      "a", "o", "u", "e", "i", "q", "y", "\u00e4",
      "\u00f6", "\u00fc", "\u00df", "s",
    });
    AFFIXES.put("FrenchLight", new String[] {
      "issement", "issant", "ement", "ive", "ficatrice", "ficateur", "catrice", "cateur",
      "atrice", "ateur", "trice", "i\u00e8me", "teuse", "teur", "euse", "\u00e8re",
      "folle", "molle", "nnelle", "nnel", "\u00e8te", "ique", "esse", "inage",
      "isation", "ual", "isateur", "ation", "ition", "ie", "x", "a",
      "u", "e", "l", "s", "r", "f", "q", "\u00e0",
      "\u00e1", "\u00e2", "\u00f4", "o", "\u00e8", "\u00e9", "\u00ea", "\u00f9",
      "\u00fb", "\u00ee", "i", "\u00e7", "c",
    });
    AFFIXES.put("FrenchMinimal", new String[] {
      "x", "a", "u", "l", "s", "r", "e", "\u00e9",
    });
    AFFIXES.put("SpanishLight", new String[] {
      "\u00e0", "\u00e1", "\u00e2", "\u00e4", "a", "\u00f2", "\u00f3", "\u00f4",
      "\u00f6", "o", "\u00e8", "\u00e9", "\u00ea", "\u00eb", "e", "\u00f9",
      "\u00fa", "\u00fb", "\u00fc", "u", "\u00ec", "\u00ed", "\u00ee", "\u00ef",
      "i", "s", "c", "z",
    });
    AFFIXES.put("SpanishMinimal", new String[] {
      "s", "\u00e0", "\u00e1", "\u00e2", "\u00e4", "a", "\u00f2", "\u00f3",
      "\u00f4", "\u00f6", "o", "\u00e8", "\u00e9", "\u00ea", "\u00eb", "e",
      "\u00f9", "\u00fa", "\u00fb", "\u00fc", "u", "\u00ec", "\u00ed", "\u00ee",
      "\u00ef", "i", "\u00f1", "n", "c", "z",
    });
    AFFIXES.put("ItalianLight", new String[] {
      "\u00e0", "\u00e1", "\u00e2", "\u00e4", "a", "\u00f2", "\u00f3", "\u00f4",
      "\u00f6", "o", "\u00e8", "\u00e9", "\u00ea", "\u00eb", "e", "\u00f9",
      "\u00fa", "\u00fb", "\u00fc", "u", "\u00ec", "\u00ed", "\u00ee", "\u00ef",
      "i", "h",
    });
    AFFIXES.put("PortugueseLight", new String[] {
      "es", "ns", "eis", "\u00e9is", "ais", "\u00f3is", "is", "\u00f5es",
      "\u00e3es", "mente", "inha", "iaca", "eira", "osa", "ica", "ida",
      "ada", "iva", "ama", "ona", "ora", "esa", "na", "a",
      "e", "o", "\u00e0", "\u00e1", "\u00e2", "\u00e4", "\u00e3", "\u00f2",
      "\u00f3", "\u00f4", "\u00f6", "\u00f5", "\u00e8", "\u00e9", "\u00ea", "\u00eb",
      "\u00f9", "\u00fa", "\u00fb", "\u00fc", "u", "\u00ec", "\u00ed", "\u00ee",
      "\u00ef", "i", "\u00e7", "c", "r", "s", "l", "z",
      "m",
    });
    AFFIXES.put("SwedishLight", new String[] {
      "elser", "heten", "arne", "erna", "ande", "else", "aste", "orna",
      "aren", "are", "ast", "het", "ar", "er", "or", "en",
      "at", "te", "et", "s", "t", "a", "e", "n",
    });
    AFFIXES.put("SwedishMinimal", new String[] {
      "arne", "erna", "arna", "orna", "aren", "are", "ar", "at",
      "er", "et", "or", "en", "s", "a", "e", "n",
    });
    AFFIXES.put("NorwegianLight", new String[] {
      "heter", "heten", "heita", "heiter", "leiken", "leikar", "dom", "het",
      "heit", "semd", "leik", "elser", "elsen", "ende", "ande", "else",
      "este", "aste", "eren", "aren", "ere", "are", "est", "ast",
      "ene", "ane", "er", "en", "et", "ar", "st", "te",
      "s", "a", "e", "n",
    });
    AFFIXES.put("NorwegianMinimal", new String[] {
      "ene", "ane", "er", "en", "et", "ar", "s", "a",
      "e",
    });
    AFFIXES.put("FinnishLight", new String[] {
      "kin", "ko", "dellinen", "dellisuus", "lla", "tse", "sti", "ni",
      "aa", "nnen", "ntena", "tten", "eiden", "neen", "niin", "seen",
      "teen", "inen", "den", "ksen", "ssa", "sta", "lta", "tta",
      "ksi", "lle", "na", "ne", "nei", "ja", "ta", "hde",
      "ei", "at", "\u00e4", "\u00e5", "a", "\u00f6", "o", "s",
      "h", "n", "k", "i", "t", "j", "e", "u",
      "p", "y",
    });
    AFFIXES.put("HungarianLight", new String[] {
      "kent", "nak", "nek", "val", "vel", "ert", "rol", "ban",
      "ben", "bol", "nal", "nel", "hoz", "hez", "tol", "al",
      "el", "at", "et", "ot", "va", "ve", "ra", "re",
      "ba", "be", "ul", "ig", "on", "en", "atok", "otok",
      "etek", "itek", "itok", "unk", "tok", "tek", "juk", "ink",
      "am", "em", "om", "ad", "ed", "od", "uk", "nk",
      "ja", "je", "im", "id", "ik", "fallthrough", "\u00e1", "a",
      "\u00eb", "\u00e9", "e", "\u00ed", "i", "\u00f3", "\u0151", "\u00f5",
      "\u00f6", "o", "\u00fa", "\u0171", "\u0169", "\u00fb", "\u00fc", "u",
      "t", "n", "m", "d", "k", "y",
    });
    AFFIXES.put("RussianLight", new String[] {
      "\u0438\u044f\u043c\u0438", "\u043e\u044f\u043c\u0438", "\u0438\u044f\u043c", "\u0438\u044f\u0445", "\u043e\u044f\u0445", "\u044f\u043c\u0438", "\u043e\u044f\u043c", "\u043e\u044c\u0432",
      "\u0430\u043c\u0438", "\u0435\u0433\u043e", "\u0435\u043c\u0443", "\u0435\u0440\u0438", "\u0438\u043c\u0438", "\u043e\u0433\u043e", "\u043e\u043c\u0443", "\u044b\u043c\u0438",
      "\u043e\u0435\u0432", "\u0430\u044f", "\u044f\u044f", "\u044f\u0445", "\u044e\u044e", "\u0430\u0445", "\u0435\u044e", "\u0438\u0445",
      "\u0438\u044f", "\u0438\u044e", "\u044c\u0432", "\u043e\u044e", "\u0443\u044e", "\u044f\u043c", "\u044b\u0445", "\u0435\u044f",
      "\u0430\u043c", "\u0435\u043c", "\u0435\u0439", "\u0451\u043c", "\u0435\u0432", "\u0438\u0439", "\u0438\u043c", "\u043e\u0435",
      "\u043e\u0439", "\u043e\u043c", "\u043e\u0432", "\u044b\u0435", "\u044b\u0439", "\u044b\u043c", "\u043c\u0438", "\u044c",
      "\u0438", "\u043d", "\u0430", "\u0435", "\u043e", "\u0443", "\u0439", "\u044b",
      "\u044f",
    });
    AFFIXES.put("Bulgarian", new String[] {
      "\u0438\u0449\u0430", "\u044f", "\u0430", "\u043e", "\u0435", "\u0435\u043d", "\u0438\u044f\u0442", "\u044a\u0442",
      "\u0442\u043e", "\u0442\u0435", "\u0442\u0430", "\u0438\u044f", "\u044f\u0442", "\u043e\u0432\u0446\u0438", "\u043e\u0432\u0435", "\u0435\u0432\u0435",
      "\u0446\u0438", "\u0437\u0438", "\u0441\u0438", "\u0438", "\u043d", "\u044a", "\u0439", "\u043a",
      "\u0433", "\u0445",
    });
    AFFIXES.put("Czech", new String[] {
      "atech", "\u011btem", "etem", "at\u016fm", "ech", "ich", "\u00edch", "\u00e9ho",
      "\u011bmi", "emi", "\u00e9mu", "\u011bte", "ete", "\u011bti", "eti", "\u00edho",
      "iho", "\u00edmi", "\u00edmu", "imu", "\u00e1ch", "ata", "aty", "\u00fdch",
      "ama", "ami", "ov\u00e9", "ovi", "\u00fdmi", "em", "es", "\u00e9m",
      "\u00edm", "\u016fm", "at", "\u00e1m", "os", "us", "\u00fdm", "mi",
      "ou", "ov", "in", "\u016fv", "\u010dt", "\u0161t", "a", "e",
      "i", "o", "u", "\u016f", "y", "\u00e1", "\u00e9", "\u00ed",
      "\u00fd", "\u011b", "c", "k", "s", "\u010d", "z", "\u017e",
      "h",
    });
    AFFIXES.put("Latvian", new String[] {
      "ajiem", "ajai", "ajam", "aj\u0101m", "ajos", "aj\u0101s", "iem", "aj\u0101",
      "ais", "ai", "ei", "\u0101m", "am", "\u0113m", "\u012bm", "im",
      "um", "us", "as", "\u0101s", "es", "os", "ij", "\u012bs",
      "\u0113s", "is", "ie", "u", "a", "i", "e", "\u0101",
      "\u0113", "\u012b", "\u016b", "o", "s", "\u0161", "k\u0161", "\u0146\u0146",
      "pj", "bj", "mj", "vj", "\u0161\u0146", "\u017e\u0146", "\u0161\u013c", "\u017e\u013c",
      "\u013c\u0146", "\u013c\u013c", "t", "n", "z", "l", "\u010d", "c",
      "\u013c", "\u0146",
    });
    AFFIXES.put("Indonesian", new String[] {
      "kah", "lah", "pun", "ku", "mu", "nya", "meng", "meny",
      "men", "mem", "me", "peng", "peny", "pen", "pem", "di",
      "ter", "ke", "ber", "belajar", "be", "per", "pelajar", "pe",
      "kan", "an", "i", "si", "a", "e", "o", "u",
      "s", "t", "r",
    });
    AFFIXES.put("Hindi", new String[] {
      "\u093e\u090f\u0902\u0917\u0940", "\u093e\u090f\u0902\u0917\u0947", "\u093e\u090a\u0902\u0917\u0940", "\u093e\u090a\u0902\u0917\u093e", "\u093e\u0907\u092f\u093e\u0901", "\u093e\u0907\u092f\u094b\u0902", "\u093e\u0907\u092f\u093e\u0902", "\u093e\u090f\u0917\u0940",
      "\u093e\u090f\u0917\u093e", "\u093e\u0913\u0917\u0940", "\u093e\u0913\u0917\u0947", "\u090f\u0902\u0917\u0940", "\u0947\u0902\u0917\u0940", "\u090f\u0902\u0917\u0947", "\u0947\u0902\u0917\u0947", "\u0942\u0902\u0917\u0940",
      "\u0942\u0902\u0917\u093e", "\u093e\u0924\u0940\u0902", "\u0928\u093e\u0913\u0902", "\u0928\u093e\u090f\u0902", "\u0924\u093e\u0913\u0902", "\u0924\u093e\u090f\u0902", "\u093f\u092f\u093e\u0901", "\u093f\u092f\u094b\u0902",
      "\u093f\u092f\u093e\u0902", "\u093e\u0915\u0930", "\u093e\u0907\u090f", "\u093e\u0908\u0902", "\u093e\u092f\u093e", "\u0947\u0917\u0940", "\u0947\u0917\u093e", "\u094b\u0917\u0940",
      "\u094b\u0917\u0947", "\u093e\u0928\u0947", "\u093e\u0928\u093e", "\u093e\u0924\u0947", "\u093e\u0924\u0940", "\u093e\u0924\u093e", "\u0924\u0940\u0902", "\u093e\u0913\u0902",
      "\u093e\u090f\u0902", "\u0941\u0913\u0902", "\u0941\u090f\u0902", "\u0941\u0906\u0902", "\u0915\u0930", "\u093e\u0913", "\u093f\u090f", "\u093e\u0908",
      "\u093e\u090f", "\u0928\u0947", "\u0928\u0940", "\u0928\u093e", "\u0924\u0947", "\u0940\u0902", "\u0924\u0940", "\u0924\u093e",
      "\u093e\u0901", "\u093e\u0902", "\u094b\u0902", "\u0947\u0902", "\u094b", "\u0947", "\u0942", "\u0941",
      "\u0940", "\u093f", "\u093e",
    });
    AFFIXES.put("HindiNormalization", new String[] {
      "\u0928", "\u094d", "\u0902", "\u0901", "\u093c", "\u0929", "\u0931", "\u0930",
      "\u0934", "\u0933", "\u0958", "\u0915", "\u0959", "\u0916", "\u095a", "\u0917",
      "\u095b", "\u091c", "\u095c", "\u0921", "\u095d", "\u0922", "\u095e", "\u092b",
      "\u095f", "\u092f", "\u200d", "\u200c", "\u0945", "\u0946", "\u0947", "\u0949",
      "\u094a", "\u094b", "\u090d", "\u090e", "\u090f", "\u0911", "\u0912", "\u0913",
      "\u0972", "\u0905", "\u0906", "\u0908", "\u0907", "\u090a", "\u0909", "\u0960",
      "\u090b", "\u0961", "\u090c", "\u0910", "\u0914", "\u0940", "\u093f", "\u0942",
      "\u0941", "\u0944", "\u0943", "\u0963", "\u0962", "\u0948", "\u094c",
    });
    AFFIXES.put("Bengali", new String[] {
      "\u09bf\u09df\u09be\u099b\u09bf\u09b2\u09be\u09ae", "\u09bf\u09a4\u09c7\u099b\u09bf\u09b2\u09be\u09ae", "\u09bf\u09a4\u09c7\u099b\u09bf\u09b2\u09c7\u09a8", "\u0987\u09a4\u09c7\u099b\u09bf\u09b2\u09c7\u09a8", "\u09bf\u09df\u09be\u099b\u09bf\u09b2\u09c7\u09a8", "\u0987\u09df\u09be\u099b\u09bf\u09b2\u09c7\u09a8", "\u09bf\u09a4\u09c7\u099b\u09bf\u09b2\u09bf", "\u09bf\u09a4\u09c7\u099b\u09bf\u09b2\u09c7",
      "\u09bf\u09df\u09be\u099b\u09bf\u09b2\u09be", "\u09bf\u09df\u09be\u099b\u09bf\u09b2\u09c7", "\u09bf\u09a4\u09c7\u099b\u09bf\u09b2\u09be", "\u09bf\u09df\u09be\u099b\u09bf\u09b2\u09bf", "\u09df\u09c7\u09a6\u09c7\u09b0\u0995\u09c7", "\u09bf\u09a4\u09c7\u099b\u09bf\u09b8", "\u09bf\u09a4\u09c7\u099b\u09c7\u09a8", "\u09bf\u09df\u09be\u099b\u09bf\u09b8",
      "\u09bf\u09df\u09be\u099b\u09c7\u09a8", "\u09c7\u099b\u09bf\u09b2\u09be\u09ae", "\u09c7\u099b\u09bf\u09b2\u09c7\u09a8", "\u09c7\u09a6\u09c7\u09b0\u0995\u09c7", "\u09bf\u09a4\u09c7\u099b\u09bf", "\u09bf\u09a4\u09c7\u099b\u09be", "\u09bf\u09a4\u09c7\u099b\u09c7", "\u099b\u09bf\u09b2\u09be\u09ae",
      "\u099b\u09bf\u09b2\u09c7\u09a8", "\u09bf\u09df\u09be\u099b\u09bf", "\u09bf\u09df\u09be\u099b\u09be", "\u09bf\u09df\u09be\u099b\u09c7", "\u09c7\u099b\u09bf\u09b2\u09c7", "\u09c7\u099b\u09bf\u09b2\u09be", "\u09df\u09c7\u09a6\u09c7\u09b0", "\u09a6\u09c7\u09b0\u0995\u09c7",
      "\u09bf\u09b2\u09be\u09ae", "\u09bf\u09b2\u09c7\u09a8", "\u09bf\u09a4\u09be\u09ae", "\u09bf\u09a4\u09c7\u09a8", "\u09bf\u09ac\u09c7\u09a8", "\u099b\u09bf\u09b2\u09bf", "\u099b\u09bf\u09b2\u09c7", "\u099b\u09bf\u09b2\u09be",
      "\u09a4\u09c7\u099b\u09c7", "\u09bf\u09a4\u09c7\u099b", "\u0996\u09be\u09a8\u09be", "\u0996\u09be\u09a8\u09bf", "\u0997\u09c1\u09b2\u09cb", "\u0997\u09c1\u09b2\u09bf", "\u09df\u09c7\u09b0\u09be", "\u09c7\u09a6\u09c7\u09b0",
      "\u09b2\u09be\u09ae", "\u09bf\u09b2\u09bf", "\u0987\u09b2\u09bf", "\u09bf\u09b2\u09c7", "\u0987\u09b2\u09c7", "\u09b2\u09c7\u09a8", "\u09bf\u09b2\u09be", "\u0987\u09b2\u09be",
      "\u09a4\u09be\u09ae", "\u09bf\u09a4\u09bf", "\u0987\u09a4\u09bf", "\u09bf\u09a4\u09c7", "\u0987\u09a4\u09c7", "\u09a4\u09c7\u09a8", "\u09bf\u09a4\u09be", "\u09bf\u09ac\u09be",
      "\u0987\u09ac\u09be", "\u09bf\u09ac\u09bf", "\u0987\u09ac\u09bf", "\u09ac\u09c7\u09a8", "\u09bf\u09ac\u09c7", "\u0987\u09ac\u09c7", "\u099b\u09c7\u09a8", "\u09df\u09cb\u09a8",
      "\u09df\u09c7\u09b0", "\u09c7\u09b0\u09be", "\u09a6\u09c7\u09b0", "\u09bf\u09b8", "\u09c7\u09a8", "\u09b2\u09bf", "\u09b2\u09c7", "\u09b2\u09be",
      "\u09a4\u09bf", "\u09a4\u09c7", "\u09a4\u09be", "\u09ac\u09bf", "\u09ac\u09c7", "\u09ac\u09be", "\u099b\u09bf", "\u099b\u09be",
      "\u099b\u09c7", "\u09c1\u09a8", "\u09c1\u0995", "\u099f\u09be", "\u099f\u09bf", "\u09a8\u09bf", "\u09c7\u09b0", "\u09b0\u09be",
      "\u0995\u09c7", "\u09bf", "\u09c0", "\u09be", "\u09cb", "\u09c7", "\u09ac", "\u09a4",
    });
    AFFIXES.put("BengaliNormalization", new String[] {
      "\u0981", "\u09c0", "\u09bf", "\u09c2", "\u09c1", "\u0995", "\u09cd", "\u0996",
      "\u0999", "\u0982", "\u09af", "\u09c7", "\u09be", "\u09ac", "\u0983", "\u09b9",
      "\u09b6", "\u09b7", "\u09b8", "\u09a3", "\u09a8", "\u09dc", "\u09dd", "\u09b0",
      "\u09ce", "\u09a4",
    });
    AFFIXES.put("Telugu", new String[] {
      "\u0c33\u0c4d\u0c33\u0c41", "\u0c21\u0c4d\u0c32\u0c41", "\u0c21\u0c41", "\u0c2e\u0c41", "\u0c35\u0c41", "\u0c32\u0c41", "\u0c28\u0c3f", "\u0c28\u0c41",
      "\u0c1a\u0c47", "\u0c15\u0c48", "\u0c32\u0c4b", "\u0c26\u0c3f", "\u0c15\u0c3f", "\u0c38\u0c41", "\u0c35\u0c48", "\u0c2a\u0c48",
      "\u0c3f", "\u0c40", "\u0c41", "\u0c42", "\u0c46", "\u0c47", "\u0c4a", "\u0c4b",
      "\u0c3e",
    });
    AFFIXES.put("TeluguNormalization", new String[] {
      "\u0c00", "\u0c01", "\u0c02", "\u0c03", "\u200d", "\u200c", "\u0c14", "\u0c13",
      "\u0c10", "\u0c0f", "\u0c06", "\u0c05", "\u0c08", "\u0c07", "\u0c0a", "\u0c09",
      "\u0c40", "\u0c3f", "\u0c42", "\u0c41", "\u0c47", "\u0c46", "\u0c4b", "\u0c4a",
      "\u0c56", "\u0c48", "\u0c12", "\u0c55", "\u0c4c",
    });
    AFFIXES.put("Sorani", new String[] {
      "\u062f\u0627", "\u0646\u0627", "\u06d5\u0648\u06d5", "\u0645\u0627\u0646", "\u06cc\u0627\u0646", "\u062a\u0627\u0646", "\u06ce\u06a9\u06cc", "\u06cc\u06d5\u06a9\u06cc",
      "\u06ce\u06a9", "\u06cc\u06d5\u06a9", "\u06d5\u06a9\u06d5", "\u06a9\u06d5", "\u06d5\u06a9\u0627\u0646", "\u06a9\u0627\u0646", "\u06cc\u0627\u0646\u06cc", "\u0627\u0646\u06cc",
      "\u0627\u0646", "\u06cc\u0627\u0646\u06d5", "\u0627\u0646\u06d5", "\u0627\u06cc\u06d5", "\u06d5\u06cc\u06d5", "\u06d5", "\u06cc",
    });
    AFFIXES.put("SoraniNormalization", new String[] {
      "\u064a", "\u0649", "\u06cc", "\u0643", "\u06a9", "\u0647", "\u06d5", "\u200c",
      "\u06be", "\u0629", "\u0631", "\u0695", "\u0692", "\u0640", "\u064b", "\u064c",
      "\u064d", "\u064e", "\u064f", "\u0650", "\u0651", "\u0652",
    });
    AFFIXES.put("Arabic", new String[] {
      "\u0627", "\u0628", "\u0629", "\u062a", "\u0641", "\u0643", "\u0644", "\u0646",
      "\u0647", "\u0648", "\u064a",
    });
    AFFIXES.put("ArabicNormalization", new String[] {
      "\u0627", "\u0622", "\u0623", "\u0625", "\u064a", "\u0649", "\u0629", "\u0647",
      "\u0640", "\u064b", "\u064c", "\u064d", "\u064e", "\u064f", "\u0650", "\u0651",
      "\u0652",
    });
    AFFIXES.put("Persian", new String[] {
      "\u0627", "\u0647", "\u062a", "\u0631", "\u0646", "\u064a", "\u200c",
    });
    AFFIXES.put("PersianNormalization", new String[] {
      "\u064a", "\u06cc", "\u06d2", "\u06a9", "\u0643", "\u0654", "\u06c0", "\u06c1",
      "\u0647",
    });
    AFFIXES.put("RomanianNormalization", new String[] {
      "\u0218", "\u0219", "\u021a", "\u021b", "\u015e", "\u015f", "\u0162", "\u0163",
    });
    AFFIXES.put("Greek", new String[] {
      "\u03ba\u03b1\u03b8\u03b5\u03c3\u03c4\u03c9\u03c4\u03bf\u03c3", "\u03ba\u03b1\u03b8\u03b5\u03c3\u03c4\u03c9\u03c4\u03c9\u03bd", "\u03b3\u03b5\u03b3\u03bf\u03bd\u03bf\u03c4\u03bf\u03c3", "\u03b3\u03b5\u03b3\u03bf\u03bd\u03bf\u03c4\u03c9\u03bd", "\u03ba\u03b1\u03b8\u03b5\u03c3\u03c4\u03c9\u03c4\u03b1", "\u03c4\u03b1\u03c4\u03bf\u03b3\u03b9\u03bf\u03c5", "\u03c4\u03b1\u03c4\u03bf\u03b3\u03b9\u03c9\u03bd", "\u03b3\u03b5\u03b3\u03bf\u03bd\u03bf\u03c4\u03b1",
      "\u03ba\u03b1\u03b8\u03b5\u03c3\u03c4\u03c9\u03c3", "\u03c3\u03ba\u03b1\u03b3\u03b9\u03bf\u03c5", "\u03c3\u03ba\u03b1\u03b3\u03b9\u03c9\u03bd", "\u03bf\u03bb\u03bf\u03b3\u03b9\u03bf\u03c5", "\u03bf\u03bb\u03bf\u03b3\u03b9\u03c9\u03bd", "\u03ba\u03c1\u03b5\u03b1\u03c4\u03bf\u03c3", "\u03ba\u03c1\u03b5\u03b1\u03c4\u03c9\u03bd", "\u03c0\u03b5\u03c1\u03b1\u03c4\u03bf\u03c3",
      "\u03c0\u03b5\u03c1\u03b1\u03c4\u03c9\u03bd", "\u03c4\u03b5\u03c1\u03b1\u03c4\u03bf\u03c3", "\u03c4\u03b5\u03c1\u03b1\u03c4\u03c9\u03bd", "\u03c4\u03b1\u03c4\u03bf\u03b3\u03b9\u03b1", "\u03b3\u03b5\u03b3\u03bf\u03bd\u03bf\u03c3", "\u03c6\u03b1\u03b3\u03b9\u03bf\u03c5", "\u03c6\u03b1\u03b3\u03b9\u03c9\u03bd", "\u03c3\u03bf\u03b3\u03b9\u03bf\u03c5",
      "\u03c3\u03bf\u03b3\u03b9\u03c9\u03bd", "\u03c3\u03ba\u03b1\u03b3\u03b9\u03b1", "\u03bf\u03bb\u03bf\u03b3\u03b9\u03b1", "\u03ba\u03c1\u03b5\u03b1\u03c4\u03b1", "\u03c0\u03b5\u03c1\u03b1\u03c4\u03b1", "\u03c4\u03b5\u03c1\u03b1\u03c4\u03b1", "\u03c6\u03b1\u03b3\u03b9\u03b1", "\u03c3\u03bf\u03b3\u03b9\u03b1",
      "\u03c6\u03c9\u03c4\u03bf\u03c3", "\u03c6\u03c9\u03c4\u03c9\u03bd", "\u03ba\u03c1\u03b5\u03b1\u03c3", "\u03c0\u03b5\u03c1\u03b1\u03c3", "\u03c4\u03b5\u03c1\u03b1\u03c3", "\u03c6\u03c9\u03c4\u03b1", "\u03c6\u03c9\u03c3", "\u03b1\u03b4\u03b5\u03c3",
      "\u03b1\u03b4\u03c9\u03bd", "\u03bf\u03ba", "\u03bc\u03b1\u03bc", "\u03bc\u03b1\u03bd", "\u03bc\u03c0\u03b1\u03bc\u03c0", "\u03c0\u03b1\u03c4\u03b5\u03c1", "\u03b3\u03b9\u03b1\u03b3\u03b9", "\u03bd\u03c4\u03b1\u03bd\u03c4",
      "\u03ba\u03c5\u03c1", "\u03b8\u03b5\u03b9", "\u03c0\u03b5\u03b8\u03b5\u03c1", "\u03b5\u03b4\u03b5\u03c3", "\u03b5\u03b4\u03c9\u03bd", "\u03bf\u03c0", "\u03b9\u03c0", "\u03b5\u03bc\u03c0",
      "\u03c5\u03c0", "\u03b3\u03b7\u03c0", "\u03b4\u03b1\u03c0", "\u03ba\u03c1\u03b1\u03c3\u03c0", "\u03bc\u03b9\u03bb", "\u03bf\u03c5\u03b4\u03b5\u03c3", "\u03bf\u03c5\u03b4\u03c9\u03bd", "\u03b1\u03c1\u03ba",
      "\u03ba\u03b1\u03bb\u03b9\u03b1\u03ba", "\u03c0\u03b5\u03c4\u03b1\u03bb", "\u03bb\u03b9\u03c7", "\u03c0\u03bb\u03b5\u03be", "\u03c3\u03ba", "\u03c3", "\u03c6\u03bb", "\u03c6\u03c1",
      "\u03b2\u03b5\u03bb", "\u03bb\u03bf\u03c5\u03bb", "\u03c7\u03bd", "\u03c3\u03c0", "\u03c4\u03c1\u03b1\u03b3", "\u03c6\u03b5", "\u03b8", "\u03b4",
      "\u03b5\u03bb", "\u03b3\u03b1\u03bb", "\u03bd", "\u03c0", "\u03b9\u03b4", "\u03c0\u03b1\u03c1", "\u03b5\u03c9\u03c3", "\u03b5\u03c9\u03bd",
      "\u03b9\u03b1", "\u03b9\u03bf\u03c5", "\u03b9\u03c9\u03bd", "\u03b1\u03bb", "\u03b1\u03b4", "\u03b5\u03bd\u03b4", "\u03b1\u03bc\u03b1\u03bd", "\u03b1\u03bc\u03bc\u03bf\u03c7\u03b1\u03bb",
      "\u03b7\u03b8", "\u03b1\u03bd\u03b7\u03b8", "\u03b1\u03bd\u03c4\u03b9\u03b4", "\u03c6\u03c5\u03c3", "\u03b2\u03c1\u03c9\u03bc", "\u03b3\u03b5\u03c1", "\u03b5\u03be\u03c9\u03b4", "\u03ba\u03b1\u03bb\u03c0",
      "\u03ba\u03b1\u03bb\u03bb\u03b9\u03bd", "\u03ba\u03b1\u03c4\u03b1\u03b4", "\u03bc\u03bf\u03c5\u03bb", "\u03bc\u03c0\u03b1\u03bd", "\u03bc\u03c0\u03b1\u03b3\u03b9\u03b1\u03c4", "\u03bc\u03c0\u03bf\u03bb", "\u03bc\u03c0\u03bf\u03c3", "\u03bd\u03b9\u03c4",
      "\u03be\u03b9\u03ba", "\u03c3\u03c5\u03bd\u03bf\u03bc\u03b7\u03bb", "\u03c0\u03b5\u03c4\u03c3", "\u03c0\u03b9\u03c4\u03c3", "\u03c0\u03b9\u03ba\u03b1\u03bd\u03c4", "\u03c0\u03bb\u03b9\u03b1\u03c4\u03c3", "\u03c0\u03bf\u03c3\u03c4\u03b5\u03bb\u03bd", "\u03c0\u03c1\u03c9\u03c4\u03bf\u03b4",
      "\u03c3\u03b5\u03c1\u03c4", "\u03c3\u03c5\u03bd\u03b1\u03b4", "\u03c4\u03c3\u03b1\u03bc", "\u03c5\u03c0\u03bf\u03b4", "\u03c6\u03b9\u03bb\u03bf\u03bd", "\u03c6\u03c5\u03bb\u03bf\u03b4", "\u03c7\u03b1\u03c3", "\u03b9\u03ba\u03b1",
      "\u03b9\u03ba\u03bf", "\u03b9\u03ba\u03bf\u03c5", "\u03b9\u03ba\u03c9\u03bd", "\u03b1\u03bd\u03b1\u03c0", "\u03b1\u03c0\u03bf\u03b8", "\u03b1\u03c0\u03bf\u03ba", "\u03b1\u03c0\u03bf\u03c3\u03c4", "\u03b2\u03bf\u03c5\u03b2",
      "\u03be\u03b5\u03b8", "\u03bf\u03c5\u03bb", "\u03c0\u03b5\u03b8", "\u03c0\u03b9\u03ba\u03c1", "\u03c0\u03bf\u03c4", "\u03c3\u03b9\u03c7", "\u03c7", "\u03b1\u03b3\u03b1\u03bc\u03b5",
      "\u03b7\u03b8\u03b7\u03ba\u03b1\u03bc\u03b5", "\u03bf\u03c5\u03c3\u03b1\u03bc\u03b5", "\u03b7\u03c3\u03b1\u03bc\u03b5", "\u03b7\u03ba\u03b1\u03bc\u03b5", "\u03b1\u03bc\u03b5", "\u03c4\u03c1", "\u03c4\u03c3", "\u03b2\u03b5\u03c4\u03b5\u03c1",
      "\u03b2\u03bf\u03c5\u03bb\u03ba", "\u03b2\u03c1\u03b1\u03c7\u03bc", "\u03b3", "\u03b4\u03c1\u03b1\u03b4\u03bf\u03c5\u03bc", "\u03ba\u03b1\u03bb\u03c0\u03bf\u03c5\u03b6", "\u03ba\u03b1\u03c3\u03c4\u03b5\u03bb", "\u03ba\u03bf\u03c1\u03bc\u03bf\u03c1", "\u03bb\u03b1\u03bf\u03c0\u03bb",
      "\u03bc\u03c9\u03b1\u03bc\u03b5\u03b8", "\u03bc", "\u03bc\u03bf\u03c5\u03c3\u03bf\u03c5\u03bb\u03bc", "\u03c0\u03b5\u03bb\u03b5\u03ba", "\u03c0\u03bb", "\u03c0\u03bf\u03bb\u03b9\u03c3", "\u03c0\u03bf\u03c1\u03c4\u03bf\u03bb", "\u03c3\u03b1\u03c1\u03b1\u03ba\u03b1\u03c4\u03c3",
      "\u03c3\u03bf\u03c5\u03bb\u03c4", "\u03c4\u03c3\u03b1\u03c1\u03bb\u03b1\u03c4", "\u03bf\u03c1\u03c6", "\u03c4\u03c3\u03b9\u03b3\u03b3", "\u03c4\u03c3\u03bf\u03c0", "\u03c6\u03c9\u03c4\u03bf\u03c3\u03c4\u03b5\u03c6", "\u03c8\u03c5\u03c7\u03bf\u03c0\u03bb", "\u03b1\u03b3",
      "\u03b4\u03b5\u03ba", "\u03b4\u03b9\u03c0\u03bb", "\u03b1\u03bc\u03b5\u03c1\u03b9\u03ba\u03b1\u03bd", "\u03bf\u03c5\u03c1", "\u03c0\u03b9\u03b8", "\u03c0\u03bf\u03c5\u03c1\u03b9\u03c4", "\u03b6\u03c9\u03bd\u03c4", "\u03b9\u03ba",
      "\u03ba\u03b1\u03c3\u03c4", "\u03ba\u03bf\u03c0", "\u03bb\u03bf\u03c5\u03b8\u03b7\u03c1", "\u03bc\u03b1\u03b9\u03bd\u03c4", "\u03bc\u03b5\u03bb", "\u03c3\u03b9\u03b3", "\u03c3\u03c4\u03b5\u03b3", "\u03c4\u03c3\u03b1\u03b3",
      "\u03c6", "\u03b5\u03c1", "\u03b1\u03b4\u03b1\u03c0", "\u03b1\u03b8\u03b9\u03b3\u03b3", "\u03b1\u03bc\u03b7\u03c7", "\u03b1\u03bd\u03b9\u03ba", "\u03b1\u03bd\u03bf\u03c1\u03b3", "\u03b1\u03c0\u03b7\u03b3",
      "\u03b1\u03c0\u03b9\u03b8", "\u03b1\u03c4\u03c3\u03b9\u03b3\u03b3", "\u03b2\u03b1\u03c3", "\u03b2\u03b1\u03c3\u03ba", "\u03b2\u03b1\u03b8\u03c5\u03b3\u03b1\u03bb", "\u03b2\u03b9\u03bf\u03bc\u03b7\u03c7", "\u03b2\u03c1\u03b1\u03c7\u03c5\u03ba", "\u03b4\u03b9\u03b1\u03c4",
      "\u03b4\u03b9\u03b1\u03c6", "\u03b5\u03bd\u03bf\u03c1\u03b3", "\u03b8\u03c5\u03c3", "\u03ba\u03b1\u03c0\u03bd\u03bf\u03b2\u03b9\u03bf\u03bc\u03b7\u03c7", "\u03ba\u03b1\u03c4\u03b1\u03b3\u03b1\u03bb", "\u03ba\u03bb\u03b9\u03b2", "\u03ba\u03bf\u03b9\u03bb\u03b1\u03c1\u03c6", "\u03bb\u03b9\u03b2",
      "\u03bc\u03b5\u03b3\u03bb\u03bf\u03b2\u03b9\u03bf\u03bc\u03b7\u03c7", "\u03bc\u03b9\u03ba\u03c1\u03bf\u03b2\u03b9\u03bf\u03bc\u03b7\u03c7", "\u03bd\u03c4\u03b1\u03b2", "\u03be\u03b7\u03c1\u03bf\u03ba\u03bb\u03b9\u03b2", "\u03bf\u03bb\u03b9\u03b3\u03bf\u03b4\u03b1\u03bc", "\u03bf\u03bb\u03bf\u03b3\u03b1\u03bb", "\u03c0\u03b5\u03bd\u03c4\u03b1\u03c1\u03c6", "\u03c0\u03b5\u03c1\u03b7\u03c6",
      "\u03c0\u03b5\u03c1\u03b9\u03c4\u03c1", "\u03c0\u03bb\u03b1\u03c4", "\u03c0\u03bf\u03bb\u03c5\u03b4\u03b1\u03c0", "\u03c0\u03bf\u03bb\u03c5\u03bc\u03b7\u03c7", "\u03c3\u03c4\u03b5\u03c6", "\u03c4\u03b1\u03b2", "\u03c4\u03b5\u03c4", "\u03c5\u03c0\u03b5\u03c1\u03b7\u03c6",
      "\u03c5\u03c0\u03bf\u03ba\u03bf\u03c0", "\u03c7\u03b1\u03bc\u03b7\u03bb\u03bf\u03b4\u03b1\u03c0", "\u03c8\u03b7\u03bb\u03bf\u03c4\u03b1\u03b2", "\u03b9\u03bf\u03c5\u03bd\u03c4\u03b1\u03bd\u03b5", "\u03b9\u03bf\u03bd\u03c4\u03b1\u03bd\u03b5", "\u03bf\u03c5\u03bd\u03c4\u03b1\u03bd\u03b5", "\u03b7\u03b8\u03b7\u03ba\u03b1\u03bd\u03b5", "\u03b9\u03bf\u03c4\u03b1\u03bd\u03b5",
      "\u03bf\u03bd\u03c4\u03b1\u03bd\u03b5", "\u03bf\u03c5\u03c3\u03b1\u03bd\u03b5", "\u03b1\u03b3\u03b1\u03bd\u03b5", "\u03b7\u03c3\u03b1\u03bd\u03b5", "\u03bf\u03c4\u03b1\u03bd\u03b5", "\u03b7\u03ba\u03b1\u03bd\u03b5", "\u03b1\u03bd\u03b5", "\u03b1\u03b2\u03b1\u03c1",
      "\u03b2\u03b5\u03bd", "\u03b5\u03bd\u03b1\u03c1", "\u03b1\u03b2\u03c1", "\u03b1\u03b8", "\u03b1\u03bd", "\u03b1\u03c0\u03bb", "\u03b2\u03b1\u03c1\u03bf\u03bd", "\u03bd\u03c4\u03c1",
      "\u03bc\u03c0\u03bf\u03c1", "\u03bd\u03b9\u03c6", "\u03c0\u03b1\u03b3", "\u03c0\u03b1\u03c1\u03b1\u03ba\u03b1\u03bb", "\u03c3\u03b5\u03c1\u03c0", "\u03c3\u03ba\u03b5\u03bb", "\u03c3\u03c5\u03c1\u03c6", "\u03c4\u03bf\u03ba",
      "\u03c5", "\u03b5\u03bc", "\u03b8\u03b1\u03c1\u03c1", "\u03b7\u03c3\u03b5\u03c4\u03b5", "\u03b5\u03c4\u03b5", "\u03bf\u03b4", "\u03b1\u03b9\u03c1", "\u03c6\u03bf\u03c1",
      "\u03c4\u03b1\u03b8", "\u03b4\u03b9\u03b1\u03b8", "\u03c3\u03c7", "\u03b5\u03c5\u03c1", "\u03c4\u03b9\u03b8", "\u03c5\u03c0\u03b5\u03c1\u03b8", "\u03c1\u03b1\u03b8", "\u03b5\u03bd\u03b8",
      "\u03c1\u03bf\u03b8", "\u03c3\u03b8", "\u03c0\u03c5\u03c1", "\u03b1\u03b9\u03bd", "\u03c3\u03c5\u03bd\u03b4", "\u03c3\u03c5\u03bd", "\u03c3\u03c5\u03bd\u03b8", "\u03c7\u03c9\u03c1",
      "\u03c0\u03bf\u03bd", "\u03b2\u03c1", "\u03ba\u03b1\u03b8", "\u03b5\u03c5\u03b8", "\u03b5\u03ba\u03b8", "\u03bd\u03b5\u03c4", "\u03c1\u03bf\u03bd", "\u03b2\u03b1\u03c1",
      "\u03b2\u03bf\u03bb", "\u03c9\u03c6\u03b5\u03bb", "\u03bf\u03bd\u03c4\u03b1\u03c3", "\u03c9\u03bd\u03c4\u03b1\u03c3", "\u03b1\u03c1\u03c7", "\u03ba\u03c1\u03b5", "\u03bf\u03bc\u03b1\u03c3\u03c4\u03b5", "\u03bf\u03bd",
      "\u03b9\u03bf\u03bc\u03b1\u03c3\u03c4\u03b5", "\u03b1\u03c0", "\u03c3\u03c5\u03bc\u03c0", "\u03b1\u03c3\u03c5\u03bc\u03c0", "\u03b1\u03ba\u03b1\u03c4\u03b1\u03c0", "\u03b1\u03bc\u03b5\u03c4\u03b1\u03bc\u03c6", "\u03b1\u03c1", "\u03b5\u03ba\u03c4\u03b5\u03bb",
      "\u03b6", "\u03be", "\u03c0\u03c1\u03bf", "\u03bd\u03b9\u03c3", "\u03b9\u03b5\u03c3\u03c4\u03b5", "\u03b5\u03c3\u03c4\u03b5", "\u03c0\u03b1\u03c1\u03b1\u03ba\u03b1\u03c4\u03b1\u03b8", "\u03c0\u03c1\u03bf\u03c3\u03b8",
      "\u03b7\u03b8\u03b7\u03ba\u03b5\u03c3", "\u03b7\u03b8\u03b7\u03ba\u03b1", "\u03b7\u03b8\u03b7\u03ba\u03b5", "\u03b7\u03ba\u03b5\u03c3", "\u03b7\u03ba\u03b1", "\u03b7\u03ba\u03b5", "\u03c3\u03ba\u03c9\u03bb", "\u03c3\u03ba\u03bf\u03c5\u03bb",
      "\u03bd\u03b1\u03c1\u03b8", "\u03c3\u03c6", "\u03bf\u03b8", "\u03c6\u03b1\u03c1\u03bc\u03b1\u03ba", "\u03c7\u03b1\u03b4", "\u03b1\u03b3\u03ba", "\u03b1\u03bd\u03b1\u03c1\u03c1", "\u03b2\u03c1\u03bf\u03bc",
      "\u03b5\u03ba\u03bb\u03b9\u03c0", "\u03bb\u03b1\u03bc\u03c0\u03b9\u03b4", "\u03bb\u03b5\u03c7", "\u03c0\u03b1\u03c4", "\u03c1", "\u03bb", "\u03bc\u03b5\u03b4", "\u03bc\u03b5\u03c3\u03b1\u03b6",
      "\u03c5\u03c0\u03bf\u03c4\u03b5\u03b9\u03bd", "\u03b1\u03bc", "\u03b1\u03b9\u03b8", "\u03b1\u03bd\u03b7\u03ba", "\u03b4\u03b5\u03c3\u03c0\u03bf\u03b6", "\u03b5\u03bd\u03b4\u03b9\u03b1\u03c6\u03b5\u03c1", "\u03b4\u03b5", "\u03b4\u03b5\u03c5\u03c4\u03b5\u03c1\u03b5\u03c5",
      "\u03ba\u03b1\u03b8\u03b1\u03c1\u03b5\u03c5", "\u03c0\u03bb\u03b5", "\u03c4\u03c3\u03b1", "\u03bf\u03c5\u03c3\u03b5\u03c3", "\u03bf\u03c5\u03c3\u03b1", "\u03bf\u03c5\u03c3\u03b5", "\u03c0\u03bf\u03b4\u03b1\u03c1", "\u03b2\u03bb\u03b5\u03c0",
      "\u03c0\u03b1\u03bd\u03c4\u03b1\u03c7", "\u03c6\u03c1\u03c5\u03b4", "\u03bc\u03b1\u03bd\u03c4\u03b9\u03bb", "\u03bc\u03b1\u03bb\u03bb", "\u03ba\u03c5\u03bc\u03b1\u03c4", "\u03bb\u03b1\u03c7", "\u03bb\u03b7\u03b3", "\u03c6\u03b1\u03b3",
      "\u03bf\u03bc", "\u03c0\u03c1\u03c9\u03c4", "\u03b1\u03b2\u03b1\u03c3\u03c4", "\u03c0\u03bf\u03bb\u03c5\u03c6", "\u03b1\u03b4\u03b7\u03c6", "\u03c0\u03b1\u03bc\u03c6", "\u03b1\u03c3\u03c0", "\u03b1\u03c6",
      "\u03b1\u03bc\u03b1\u03bb", "\u03b1\u03bc\u03b1\u03bb\u03bb\u03b9", "\u03b1\u03bd\u03c5\u03c3\u03c4", "\u03b1\u03c0\u03b5\u03c1", "\u03b1\u03c3\u03c0\u03b1\u03c1", "\u03b1\u03c7\u03b1\u03c1", "\u03b4\u03b5\u03c1\u03b2\u03b5\u03bd", "\u03b4\u03c1\u03bf\u03c3\u03bf\u03c0",
      "\u03be\u03b5\u03c6", "\u03bd\u03b5\u03bf\u03c0", "\u03bd\u03bf\u03bc\u03bf\u03c4", "\u03bf\u03bb\u03bf\u03c0", "\u03bf\u03bc\u03bf\u03c4", "\u03c0\u03c1\u03bf\u03c3\u03c4", "\u03c0\u03c1\u03bf\u03c3\u03c9\u03c0\u03bf\u03c0", "\u03c3\u03c5\u03bd\u03c4",
      "\u03c4", "\u03c5\u03c0\u03bf\u03c4", "\u03c7\u03b1\u03c1", "\u03b1\u03b5\u03b9\u03c0", "\u03b1\u03b9\u03bc\u03bf\u03c3\u03c4", "\u03b1\u03bd\u03c5\u03c0", "\u03b1\u03c0\u03bf\u03c4", "\u03b1\u03c1\u03c4\u03b9\u03c0",
      "\u03b5\u03bd", "\u03b5\u03c0\u03b9\u03c4", "\u03ba\u03c1\u03bf\u03ba\u03b1\u03bb\u03bf\u03c0", "\u03c3\u03b9\u03b4\u03b7\u03c1\u03bf\u03c0", "\u03bd\u03b1\u03c5", "\u03bf\u03c5\u03bb\u03b1\u03bc", "\u03c8\u03bf\u03c6", "\u03bd\u03b1\u03c5\u03bb\u03bf\u03c7",
      "\u03b1\u03b3\u03b5\u03c3", "\u03b1\u03b3\u03b1", "\u03b1\u03b3\u03b5", "\u03bf\u03c6", "\u03c0\u03b5\u03bb", "\u03c7\u03bf\u03c1\u03c4", "\u03bb\u03bb", "\u03c1\u03c0",
      "\u03c0\u03c1", "\u03bb\u03bf\u03c7", "\u03c3\u03bc\u03b7\u03bd", "\u03ba\u03bf\u03bb\u03bb", "\u03c7\u03b5\u03c1\u03c3\u03bf\u03bd", "\u03b4\u03c9\u03b4\u03b5\u03ba\u03b1\u03bd", "\u03b5\u03c1\u03b7\u03bc\u03bf\u03bd", "\u03bc\u03b5\u03b3\u03b1\u03bb\u03bf\u03bd",
      "\u03b5\u03c0\u03c4\u03b1\u03bd", "\u03b7\u03c3\u03bf\u03c5", "\u03b7\u03c3\u03b5", "\u03b7\u03c3\u03b1", "\u03b1\u03c3\u03b2", "\u03c3\u03b2", "\u03b1\u03c7\u03c1", "\u03c7\u03c1",
      "\u03b1\u03b5\u03b9\u03bc\u03bd", "\u03b4\u03c5\u03c3\u03c7\u03c1", "\u03b5\u03c5\u03c7\u03c1", "\u03ba\u03bf\u03b9\u03bd\u03bf\u03c7\u03c1", "\u03c0\u03b1\u03bb\u03b9\u03bc\u03c8", "\u03b7\u03c3\u03c4\u03b5", "\u03c3\u03c0\u03b9", "\u03c3\u03c4\u03c1\u03b1\u03b2\u03bf\u03bc\u03bf\u03c5\u03c4\u03c3",
      "\u03ba\u03b1\u03ba\u03bf\u03bc\u03bf\u03c5\u03c4\u03c3", "\u03b5\u03be\u03c9\u03bd", "\u03b7\u03c3\u03bf\u03c5\u03bd\u03b5", "\u03b7\u03b8\u03bf\u03c5\u03bd\u03b5", "\u03bf\u03c5\u03bd\u03b5", "\u03c0\u03b1\u03c1\u03b1\u03c3\u03bf\u03c5\u03c3", "\u03c9\u03c1\u03b9\u03bf\u03c0\u03bb", "\u03b1\u03b6",
      "\u03b1\u03bb\u03bb\u03bf\u03c3\u03bf\u03c5\u03c3", "\u03b1\u03c3\u03bf\u03c5\u03c3", "\u03b7\u03c3\u03bf\u03c5\u03bc\u03b5", "\u03b7\u03b8\u03bf\u03c5\u03bc\u03b5", "\u03bf\u03c5\u03bc\u03b5", "\u03bc\u03b1\u03c4\u03c9\u03bd", "\u03bc\u03b1\u03c4\u03bf\u03c3", "\u03bc\u03b1\u03c4\u03b1",
      "\u03b9\u03bf\u03bd\u03c4\u03bf\u03c5\u03c3\u03b1\u03bd", "\u03b9\u03bf\u03bc\u03b1\u03c3\u03c4\u03b1\u03bd", "\u03b9\u03bf\u03c3\u03b1\u03c3\u03c4\u03b1\u03bd", "\u03b9\u03bf\u03c5\u03bc\u03b1\u03c3\u03c4\u03b5", "\u03bf\u03bd\u03c4\u03bf\u03c5\u03c3\u03b1\u03bd", "\u03b9\u03b5\u03bc\u03b1\u03c3\u03c4\u03b5", "\u03b9\u03b5\u03c3\u03b1\u03c3\u03c4\u03b5", "\u03b9\u03bf\u03bc\u03bf\u03c5\u03bd\u03b1",
      "\u03b9\u03bf\u03c3\u03b1\u03c3\u03c4\u03b5", "\u03b9\u03bf\u03c3\u03bf\u03c5\u03bd\u03b1", "\u03b9\u03bf\u03c5\u03bd\u03c4\u03b1\u03b9", "\u03b9\u03bf\u03c5\u03bd\u03c4\u03b1\u03bd", "\u03b7\u03b8\u03b7\u03ba\u03b1\u03c4\u03b5", "\u03bf\u03bc\u03b1\u03c3\u03c4\u03b1\u03bd", "\u03bf\u03c3\u03b1\u03c3\u03c4\u03b1\u03bd", "\u03bf\u03c5\u03bc\u03b1\u03c3\u03c4\u03b5",
      "\u03b9\u03bf\u03bc\u03bf\u03c5\u03bd", "\u03b9\u03bf\u03bd\u03c4\u03b1\u03bd", "\u03b9\u03bf\u03c3\u03bf\u03c5\u03bd", "\u03b7\u03b8\u03b5\u03b9\u03c4\u03b5", "\u03b7\u03b8\u03b7\u03ba\u03b1\u03bd", "\u03bf\u03bc\u03bf\u03c5\u03bd\u03b1", "\u03bf\u03c3\u03b1\u03c3\u03c4\u03b5", "\u03bf\u03c3\u03bf\u03c5\u03bd\u03b1",
      "\u03bf\u03c5\u03bd\u03c4\u03b1\u03b9", "\u03bf\u03c5\u03bd\u03c4\u03b1\u03bd", "\u03bf\u03c5\u03c3\u03b1\u03c4\u03b5", "\u03b1\u03b3\u03b1\u03c4\u03b5", "\u03b9\u03b5\u03bc\u03b1\u03b9", "\u03b9\u03b5\u03c4\u03b1\u03b9", "\u03b9\u03b5\u03c3\u03b1\u03b9", "\u03b9\u03bf\u03c4\u03b1\u03bd",
      "\u03b9\u03bf\u03c5\u03bc\u03b1", "\u03b7\u03b8\u03b5\u03b9\u03c3", "\u03b7\u03b8\u03bf\u03c5\u03bd", "\u03b7\u03ba\u03b1\u03c4\u03b5", "\u03b7\u03c3\u03b1\u03c4\u03b5", "\u03b7\u03c3\u03bf\u03c5\u03bd", "\u03bf\u03bc\u03bf\u03c5\u03bd", "\u03bf\u03bd\u03c4\u03b1\u03b9",
      "\u03bf\u03bd\u03c4\u03b1\u03bd", "\u03bf\u03c3\u03bf\u03c5\u03bd", "\u03bf\u03c5\u03bc\u03b1\u03b9", "\u03bf\u03c5\u03c3\u03b1\u03bd", "\u03b1\u03b3\u03b1\u03bd", "\u03b1\u03bc\u03b1\u03b9", "\u03b1\u03c3\u03b1\u03b9", "\u03b1\u03c4\u03b1\u03b9",
      "\u03b5\u03b9\u03c4\u03b5", "\u03b5\u03c3\u03b1\u03b9", "\u03b5\u03c4\u03b1\u03b9", "\u03b7\u03b4\u03b5\u03c3", "\u03b7\u03b4\u03c9\u03bd", "\u03b7\u03b8\u03b5\u03b9", "\u03b7\u03ba\u03b1\u03bd", "\u03b7\u03c3\u03b1\u03bd",
      "\u03b7\u03c3\u03b5\u03b9", "\u03b7\u03c3\u03b5\u03c3", "\u03bf\u03bc\u03b1\u03b9", "\u03bf\u03c4\u03b1\u03bd", "\u03b1\u03b5\u03b9", "\u03b5\u03b9\u03c3", "\u03b7\u03b8\u03c9", "\u03b7\u03c3\u03c9",
      "\u03bf\u03c5\u03bd", "\u03bf\u03c5\u03c3", "\u03b1\u03c3", "\u03b1\u03c9", "\u03b5\u03b9", "\u03b5\u03c3", "\u03b7\u03c3", "\u03bf\u03b9",
      "\u03bf\u03c3", "\u03bf\u03c5", "\u03c5\u03c3", "\u03c9\u03bd", "\u03b5\u03c3\u03c4\u03b5\u03c1", "\u03b5\u03c3\u03c4\u03b1\u03c4", "\u03bf\u03c4\u03b5\u03c1", "\u03bf\u03c4\u03b1\u03c4",
      "\u03c5\u03c4\u03b5\u03c1", "\u03c5\u03c4\u03b1\u03c4", "\u03c9\u03c4\u03b5\u03c1", "\u03c9\u03c4\u03b1\u03c4", "\u03b1", "\u03bf", "\u03c9", "\u03b5",
      "\u03b7", "\u03b9",
    });
    AFFIXES.put("GreekLowerCase", new String[] {
      "\u03c2", "\u03c3", "\u0386", "\u03ac", "\u03b1", "\u0388", "\u03ad", "\u03b5",
      "\u0389", "\u03ae", "\u03b7", "\u038a", "\u03aa", "\u03af", "\u03ca", "\u0390",
      "\u03b9", "\u038e", "\u03ab", "\u03cd", "\u03cb", "\u03b0", "\u03c5", "\u038c",
      "\u03cc", "\u03bf", "\u038f", "\u03ce", "\u03c9", "\u03a2",
    });
    AFFIXES.put("SpanishPlural", new String[] {
      "abrebotellas", "abrecartas", "abrelatas", "afueras", "albatros", "albricias", "aleda\u00f1os", "alexis",
      "alicates", "analisis", "andurriales", "antitesis", "a\u00f1icos", "apendicitis", "apocalipsis", "arcoiris",
      "aries", "bilis", "boletus", "boris", "brindis", "cactus", "canutas", "caries",
      "cascanueces", "cascarrabias", "ciempies", "cifosis", "cortaplumas", "corpus", "cosmos", "cosquillas",
      "creces", "crisis", "cuatrocientas", "cuatrocientos", "cuelgacapas", "cuentacuentos", "cuentapasos", "cumplea\u00f1os",
      "doscientas", "doscientos", "dosis", "enseres", "entonces", "esponsales", "estatus", "exequias",
      "fauces", "forceps", "fotosintesis", "gafas", "gafotas", "gargaras", "gris", "honorarios",
      "ictus", "jueves", "lapsus", "lavacoches", "lavaplatos", "limpiabotas", "lunes", "maitines",
      "martes", "mondadientes", "novecientas", "novecientos", "nupcias", "ochocientas", "ochocientos", "pais",
      "paris", "parabrisas", "paracaidas", "parachoques", "paraguas", "pararrayos", "pisapapeles", "piscis",
      "portaaviones", "portamaletas", "portamantas", "quinientas", "quinientos", "quitamanchas", "recogepelotas", "rictus",
      "rompeolas", "sacacorchos", "sacapuntas", "saltamontes", "salvavidas", "seis", "seiscientas", "seiscientos",
      "setecientas", "setecientos", "sintesis", "tenis", "tifus", "trabalenguas", "vacaciones", "venus",
      "versus", "viacrucis", "virus", "viveres", "volandas", "yoes", "noes", "sies",
      "clubes", "faralaes", "albalaes", "itemes", "albumes", "sandwiches", "relojes", "bojes",
      "contrarreloj", "carcajes", "s", "q", "g", "u", "i", "e",
      "r", "d", "l", "n", "x", "y", "t", "c",
      "z", "a", "o", "\u00e0", "\u00e1", "\u00e2", "\u00e4", "\u00f2",
      "\u00f3", "\u00f4", "\u00f6", "\u00e8", "\u00e9", "\u00ea", "\u00eb", "\u00f9",
      "\u00fa", "\u00fb", "\u00fc", "\u00ec", "\u00ed", "\u00ee", "\u00ef",
    });
    AFFIXES.put("Brazilian", new String[] {
      "pt-BR", ";", "a", "e", "i", "o", "u", "c",
      "n", "uciones", "imentos", "amentos", "adores", "adoras", "logias", "log",
      "encias", "ente", "amente", "idades", "acoes", "imento", "amento", "adora",
      "ismos", "istas", "logia", "ucion", "encia", "mente", "idade", "acao",
      "ezas", "icos", "icas", "ismo", "avel", "ivel", "ista", "osos",
      "osas", "ador", "ivas", "ivos", "iras", "ir", "eza", "ico",
      "ica", "oso", "osa", "iva", "ivo", "ira", "issemos", "essemos",
      "assemos", "ariamos", "eriamos", "iriamos", "iremos", "eremos", "aremos", "avamos",
      "iramos", "eramos", "aramos", "asseis", "esseis", "isseis", "arieis", "erieis",
      "irieis", "irmos", "iamos", "armos", "ermos", "areis", "ereis", "ireis",
      "asses", "esses", "isses", "astes", "assem", "essem", "issem", "ardes",
      "erdes", "irdes", "ariam", "eriam", "iriam", "arias", "erias", "irias",
      "estes", "istes", "aveis", "aria", "eria", "iria", "asse", "esse",
      "isse", "aste", "este", "iste", "arei", "erei", "irei", "aram",
      "eram", "iram", "avam", "arem", "erem", "irem", "ando", "endo",
      "indo", "arao", "erao", "irao", "adas", "idas", "aras", "eras",
      "avas", "ares", "eres", "ires", "ados", "idos", "amos", "emos",
      "imos", "ieis", "ada", "ida", "ara", "era", "ava", "iam",
      "ado", "ido", "ias", "ais", "eis", "ear", "ia", "ei",
      "am", "em", "ar", "er", "as", "es", "is", "eu",
      "iu", "ou", "os", "gu", "ci", ")", "\u00e1", "\u00e2",
      "\u00e3", "\u00e9", "\u00ea", "\u00ed", "\u00f3", "\u00f4", "\u00f5", "\u00fa",
      "\u00fc", "\u00e7", "\u00f1", "\"", "-", ",", ".", "?",
      "!",
    });
    AFFIXES.put("SerbianNormalization", new String[] {
      "\u0430", "a", "\u0431", "b", "\u0432", "v", "\u0433", "g",
      "\u0434", "d", "\u0452", "\u0111", "j", "\u0435", "e", "\u0436",
      "\u0437", "\u017e", "z", "\u0438", "i", "\u0458", "\u043a", "k",
      "\u043b", "l", "\u0459", "\u043c", "m", "\u043d", "n", "\u045a",
      "\u043e", "o", "\u043f", "p", "\u0440", "r", "\u0441", "s",
      "\u0442", "t", "\u045b", "\u0446", "\u0447", "\u010d", "\u0107", "c",
      "\u0443", "u", "\u0444", "f", "\u0445", "h", "\u045f", "\u0448",
      "\u0161",
    });
    AFFIXES.put("SerbianNormalizationRegular", new String[] {
      "\u0430", "a", "\u0431", "b", "\u0432", "v", "\u0433", "g",
      "\u0434", "d", "\u0452", "\u0111", "\u0435", "e", "\u0436", "\u017e",
      "\u0437", "z", "\u0438", "i", "\u0458", "j", "\u043a", "k",
      "\u043b", "l", "\u0459", "\u043c", "m", "\u043d", "n", "\u045a",
      "\u043e", "o", "\u043f", "p", "\u0440", "r", "\u0441", "s",
      "\u0442", "t", "\u045b", "\u0107", "\u0443", "u", "\u0444", "f",
      "\u0445", "h", "\u0446", "c", "\u0447", "\u010d", "\u045f", "\u0448",
      "\u0161",
    });
    AFFIXES.put("TurkishLowerCase", new String[] {
      "I", "i", "\u0131", "\u0307",
    });
    AFFIXES.put("Portuguese", new String[] {
      "ns", "\u00f5es", "\u00e3es", "ais", "\u00e9is", "eis", "\u00f3is", "is",
      "les", "res", "s", "mente", "ona", "\u00e3", "ora", "na",
      "inha", "esa", "osa", "\u00edaca", "ica", "ada", "ida", "\u00edda",
      "ima", "iva", "eira", "d\u00edssimo", "abil\u00edssimo", "\u00edssimo", "\u00e9simo", "\u00e9rrimo",
      "zinho", "quinho", "uinho", "adinho", "inho", "alh\u00e3o", "u\u00e7a", "a\u00e7o",
      "a\u00e7a", "ad\u00e3o", "id\u00e3o", "\u00e1zio", "arraz", "zarr\u00e3o", "arr\u00e3o", "arra",
      "z\u00e3o", "\u00e3o", "encialista", "alista", "agem", "iamento", "amento", "imento",
      "mento", "alizado", "atizado", "tizado", "izado", "ativo", "tivo", "ivo",
      "ado", "ido", "ador", "edor", "idor", "dor", "sor", "atoria",
      "tor", "or", "abilidade", "icionista", "cionista", "ionista", "ionar", "ional",
      "\u00eancia", "\u00e2ncia", "edouro", "queiro", "adeiro", "eiro", "uoso", "oso",
      "aliza\u00e7", "atiza\u00e7", "tiza\u00e7", "iza\u00e7", "a\u00e7", "i\u00e7", "\u00e1rio", "at\u00f3rio",
      "rio", "\u00e9rio", "\u00eas", "eza", "ez", "esco", "ante", "\u00e1stico",
      "al\u00edstico", "\u00e1utico", "\u00eautico", "tico", "ico", "ividade", "idade", "oria",
      "encial", "ista", "auta", "quice", "ice", "\u00edaco", "ente", "ense",
      "inal", "ano", "\u00e1vel", "\u00edvel", "vel", "bil", "ura", "ural",
      "ual", "ial", "al", "alismo", "ivismo", "ismo", "ar\u00edamo", "\u00e1ssemo",
      "er\u00edamo", "\u00eassemo", "ir\u00edamo", "\u00edssemo", "\u00e1ramo", "\u00e1rei", "aremo", "ariam",
      "ar\u00edei", "\u00e1ssei", "assem", "\u00e1vamo", "\u00earamo", "eremo", "eriam", "er\u00edei",
      "\u00eassei", "essem", "\u00edramo", "iremo", "iriam", "ir\u00edei", "\u00edssei", "issem",
      "ando", "endo", "indo", "ondo", "aram", "ar\u00e3o", "arde", "arei",
      "arem", "aria", "armo", "asse", "aste", "avam", "\u00e1vei", "eram",
      "er\u00e3o", "erde", "erei", "\u00earei", "erem", "eria", "ermo", "esse",
      "este", "\u00edamo", "iram", "\u00edram", "ir\u00e3o", "irde", "irei", "irem",
      "iria", "irmo", "isse", "iste", "iava", "amo", "iona", "ara",
      "ar\u00e1", "are", "ava", "emo", "era", "er\u00e1", "ere", "iam",
      "\u00edei", "imo", "ira", "\u00eddo", "ir\u00e1", "tizar", "izar", "itar",
      "ire", "omo", "ai", "am", "ear", "ar", "uei", "u\u00eda",
      "ei", "guem", "em", "er", "eu", "ia", "ir", "iu",
      "eou", "ou", "i", "gue", "\u00e1", "\u00ea", "a", "e",
      "o",
    });
    AFFIXES.put("Galician", new String[] {
      "ns", "\u00f3s", "\u00f5es", "\u00e3es", "ais", "\u00e1is", "\u00e9is", "eis",
      "\u00f3is", "ois", "\u00eds", "is", "les", "res", "ces", "zes",
      "ises", "\u00e1s", "ses", "s", "\u00edssimo", "\u00edssima", "a\u00e7o", "a\u00e7a",
      "u\u00e7a", "lhar", "lher", "lhor", "lho", "nhar", "nhor", "nho",
      "nha", "\u00e1rio", "\u00e1ria", "able", "\u00e1vel", "ible", "\u00edvel", "\u00e7om",
      "agem", "age", "\u00e3o", "ao", "au", "om", "m", "mente",
      "d\u00edsimo", "d\u00edsima", "bil\u00edsimo", "bil\u00edsima", "\u00edsimo", "\u00edsima", "\u00e9simo", "\u00e9sima",
      "\u00e9rrimo", "\u00e9rrima", "ana", "\u00e1n", "azo", "aza", "allo", "alla",
      "arra", "astro", "astra", "\u00e1zio", "elo", "eta", "ete", "ica",
      "ico", "exo", "exa", "id\u00e3o", "i\u00f1o", "i\u00f1a", "ito", "ita",
      "oide", "ola", "olo", "ote", "ota", "cho", "cha", "uco",
      "uzo", "uza", "uxa", "uxo", "ello", "ella", "dade", "ificar",
      "eiro", "eira", "ario", "aria", "\u00edstico", "ista", "ado", "ato",
      "ido", "ida", "\u00edda", "udo", "uda", "ada", "dela", "ela",
      "\u00e1bel", "\u00edbel", "nte", "ncia", "nza", "acia", "icia", "iza",
      "exar", "aci\u00f3n", "ici\u00f3n", "ci\u00f3n", "si\u00f3n", "az\u00f3n", "\u00f3n", "ona",
      "oa", "aco", "aca", "al", "dor", "tor", "or", "ora",
      "ar\u00eda", "axe", "dizo", "eza", "ez", "engo", "ego", "oso",
      "osa", "ume", "ura", "i\u00f1ar", "il", "esco", "isco", "ivo",
      "aba", "abade", "\u00e1bade", "abamo", "\u00e1bamo", "aban", "ache", "ade",
      "an", "ando", "ar", "arade", "aramo", "ar\u00e1n", "aran", "\u00e1rade",
      "ariade", "ar\u00edade", "arian", "ariamo", "aron", "ase", "asede", "\u00e1sede",
      "asemo", "\u00e1semo", "asen", "avan", "ar\u00edamo", "assen", "\u00e1ssemo", "er\u00edamo",
      "\u00eassemo", "ir\u00edamo", "\u00edssemo", "\u00e1ramo", "\u00e1rei", "aren", "aremo", "ar\u00edei",
      "\u00e1ssei", "\u00e1vamo", "\u00earamo", "eremo", "er\u00edei", "\u00eassei", "\u00edramo", "iremo",
      "ir\u00edei", "\u00edssei", "issen", "endo", "indo", "ondo", "arde", "arei",
      "armo", "asse", "aste", "\u00e1vei", "er\u00e3o", "erde", "erei", "\u00earei",
      "eren", "eria", "ermo", "este", "\u00edamo", "ian", "irde", "irei",
      "iren", "iria", "irmo", "isse", "iste", "iava", "amo", "iona",
      "ara", "ar\u00e1", "are", "ava", "emo", "era", "er\u00e1", "ere",
      "\u00edei", "in", "imo", "ira", "\u00eddo", "ir\u00e1", "tizar", "izar",
      "itar", "ire", "omo", "ai", "ear", "uei", "u\u00eda", "ei",
      "er", "eu", "ia", "ir", "iu", "eou", "ou", "i",
      "ede", "en", "erade", "\u00e9rade", "eran", "eramo", "\u00e9ramo", "er\u00e1n",
      "er\u00eda", "eriade", "er\u00edade", "eriamo", "erian", "er\u00edan", "eron", "ese",
      "esedes", "\u00e9sedes", "esemo", "\u00e9semo", "esen", "\u00eassede", "\u00eda", "iade",
      "\u00edade", "iamo", "\u00edan", "iche", "ide", "irade", "\u00edrade", "iramo",
      "ir\u00e1n", "ir\u00eda", "iriade", "ir\u00edade", "iriamo", "irian", "ir\u00edan", "iron",
      "ise", "isede", "\u00edsede", "isemo", "\u00edsemo", "isen", "\u00edssede", "gue",
      "que", "a", "e", "o", "\u00e2", "\u00e3", "\u00ea", "\u00f4",
      "\u00e1", "\u00e9", "\u00f3",
    });
    AFFIXES.put("PortugueseExceptions", new String[] {
      "m\u00e3es", "cais", "mais", "l\u00e1pis", "cr\u00facis", "biqu\u00ednis", "pois", "depois",
      "dois", "leis", "\u00e1rvores", "ali\u00e1s", "pires", "mas", "menos", "f\u00e9rias",
      "fezes", "p\u00easames", "g\u00e1s", "atr\u00e1s", "mois\u00e9s", "atrav\u00e9s", "conv\u00e9s", "\u00eas",
      "pa\u00eds", "ap\u00f3s", "ambas", "ambos", "messias", "experimente", "abandona", "lona",
      "iona", "cortisona", "mon\u00f3tona", "maratona", "acetona", "detona", "carona", "amanh\u00e3",
      "arapu\u00e3", "f\u00e3", "div\u00e3", "guiana", "campana", "grana", "caravana", "banana",
      "paisana", "rainha", "linha", "minha", "mesa", "obesa", "princesa", "turquesa",
      "ilesa", "pesa", "presa", "mucosa", "prosa", "dica", "pitada", "vida",
      "d\u00favida", "reca\u00edda", "sa\u00edda", "v\u00edtima", "saliva", "oliva", "beira", "cadeira",
      "frigideira", "bandeira", "feira", "capoeira", "barreira", "fronteira", "besteira", "poeira",
      "caminho", "cominho", "antebra\u00e7o", "top\u00e1zio", "coaliz\u00e3o", "camar\u00e3o", "chimarr\u00e3o", "can\u00e7\u00e3o",
      "cora\u00e7\u00e3o", "embri\u00e3o", "grot\u00e3o", "glut\u00e3o", "fic\u00e7\u00e3o", "fog\u00e3o", "fei\u00e7\u00e3o", "furac\u00e3o",
      "gam\u00e3o", "lampi\u00e3o", "le\u00e3o", "macac\u00e3o", "na\u00e7\u00e3o", "\u00f3rf\u00e3o", "org\u00e3o", "patr\u00e3o",
      "port\u00e3o", "quinh\u00e3o", "rinc\u00e3o", "tra\u00e7\u00e3o", "falc\u00e3o", "espi\u00e3o", "mam\u00e3o", "foli\u00e3o",
      "cord\u00e3o", "aptid\u00e3o", "campe\u00e3o", "colch\u00e3o", "lim\u00e3o", "leil\u00e3o", "mel\u00e3o", "bar\u00e3o",
      "milh\u00e3o", "bilh\u00e3o", "fus\u00e3o", "crist\u00e3o", "ilus\u00e3o", "capit\u00e3o", "esta\u00e7\u00e3o", "sen\u00e3o",
      "coragem", "chantagem", "vantagem", "carruagem", "firmamento", "fundamento", "departamento", "elemento",
      "complemento", "instrumento", "alfabetizado", "organizado", "pulverizado", "pejorativo", "relativo", "passivo",
      "possessivo", "positivo", "grado", "c\u00e2ndido", "consolido", "r\u00e1pido", "decido", "t\u00edmido",
      "duvido", "marido", "ouvidor", "assessor", "benfeitor", "leitor", "editor", "pastor",
      "produtor", "promotor", "consultor", "motor", "melhor", "redor", "rigor", "sensor",
      "tambor", "tumor", "terior", "favor", "autor", "ambul\u00e2ncia", "desfiladeiro", "pioneiro",
      "mosteiro", "precioso", "organiza\u00e7", "equa\u00e7", "rela\u00e7", "elei\u00e7", "volunt\u00e1rio", "sal\u00e1rio",
      "anivers\u00e1rio", "di\u00e1rio", "lion\u00e1rio", "arm\u00e1rio", "compuls\u00f3rio", "pr\u00f3prio", "st\u00e9rio", "gigante",
      "elefante", "adiante", "possante", "instante", "restaurante", "eclesi\u00e1stico", "pol\u00edtico", "diagnostico",
      "pr\u00e1tico", "dom\u00e9stico", "diagn\u00f3stico", "id\u00eantico", "alop\u00e1tico", "art\u00edstico", "aut\u00eantico", "ecl\u00e9tico",
      "cr\u00edtico", "critico", "tico", "p\u00fablico", "explico", "autoridade", "comunidade", "categoria",
      "c\u00famplice", "freq\u00fcente", "alimente", "acrescente", "permanente", "oriente", "aparente", "af\u00e1vel",
      "razo\u00e1vel", "pot\u00e1vel", "vulner\u00e1vel", "poss\u00edvel", "sol\u00favel", "imatura", "acupuntura", "costura",
      "bissexual", "virtual", "visual", "pontual", "afinal", "animal", "estatal", "desleal",
      "fiscal", "formal", "pessoal", "liberal", "postal", "sideral", "sucursal", "cinismo",
      "agravam", "faroeste", "agreste", "admirei", "adquirem", "ampliava", "arara", "prepara",
      "alvar\u00e1", "prepare", "agrava", "acelera", "espera", "espere", "enfiam", "ampliam",
      "elogiam", "ensaiam", "reprimo", "intimo", "\u00edntimo", "nimo", "queimo", "ximo",
      "s\u00e1tira", "alfabetizar", "organizar", "acreditar", "explicitar", "estreitar", "adquire", "alardear",
      "nuclear", "azar", "bazaar", "patamar", "alem", "virgem", "\u00e9ter", "pier",
      "chapeu", "est\u00f3ria", "fatia", "acia", "praia", "elogia", "mania", "l\u00e1bia",
      "aprecia", "pol\u00edcia", "arredia", "cheia", "\u00e1sia", "freir", "i", "gangue",
      "jegue", "beb\u00ea", "\u00e3o",
    });
    AFFIXES.put("GalicianExceptions", new String[] {
      "luns", "furatap\u00f3ns", "furatapons", "m\u00e3es", "magalh\u00e3es", "cais", "tais", "mais",
      "pais", "ademais", "c\u00e1is", "t\u00e1is", "m\u00e1is", "p\u00e1is", "adem\u00e1is", "escornab\u00f3is",
      "escornabois", "pa\u00eds", "menfis", "kinguis", "ingles", "marselles", "montreales", "senegales",
      "manizales", "m\u00f3stoles", "n\u00e1poles", "petres", "henares", "c\u00e1ceres", "baleares", "linares",
      "londres", "mieres", "miraflores", "m\u00e9rcores", "venres", "pires", "m\u00e1s", "barbad\u00e9s",
      "barcelon\u00e9s", "canton\u00e9s", "gabon\u00e9s", "llan\u00e9s", "medin\u00e9s", "escoc\u00e9s", "escoc\u00eas", "franc\u00eas",
      "barcelon\u00eas", "canton\u00eas", "macram\u00e9s", "reves", "barcelones", "cantones", "gabones", "llanes",
      "magallanes", "medines", "escoces", "frances", "xoves", "martes", "ali\u00e1s", "l\u00e1pis",
      "mas", "menos", "f\u00e9rias", "p\u00easames", "cr\u00facis", "cangas", "atenas", "asturias",
      "canarias", "filipinas", "honduras", "molucas", "caldas", "mascare\u00f1as", "micenas", "covarrubias",
      "psoas", "\u00f3culos", "nupcias", "m", "n", "experimente", "vehemente", "sedimente",
      "argana", "banana", "choupana", "espadana", "faciana", "iguana", "lantana", "macana",
      "membrana", "mesana", "nirvana", "obsidiana", "palangana", "pavana", "persiana", "pestana",
      "porcelana", "pseudomembrana", "roldana", "s\u00e1bana", "salangana", "saragana", "ventana", "adem\u00e1n",
      "bard\u00e1n", "barreg\u00e1n", "corric\u00e1n", "curric\u00e1n", "fais\u00e1n", "furac\u00e1n", "fust\u00e1n", "gab\u00e1n",
      "gabi\u00e1n", "gal\u00e1n", "ga\u00f1\u00e1n", "lavac\u00e1n", "maz\u00e1n", "mour\u00e1n", "rabad\u00e1n", "ser\u00e1n",
      "serr\u00e1n", "tab\u00e1n", "tit\u00e1n", "tobog\u00e1n", "ver\u00e1n", "volc\u00e1n", "volov\u00e1n", "abrazo",
      "espazo", "andazo", "bagazo", "balazo", "bandazo", "cachazo", "carazo", "denazo",
      "engazo", "famazo", "lampreazo", "pantocazo", "pedazo", "pre\u00f1azo", "regazo", "ribazo",
      "sobrazo", "terrazo", "trompazo", "alcarraza", "ameaza", "baraza", "broucaza", "burgaza",
      "cabaza", "cachaza", "calaza", "carpaza", "carraza", "coiraza", "colmaza", "fogaza",
      "famaza", "labaza", "li\u00f1aza", "melaza", "mordaza", "paraza", "pinaza", "rabaza",
      "rapaza", "trancaza", "traballo", "cigarra", "cinzarra", "balastro", "bimbastro", "canastro",
      "retropilastro", "banastra", "canastra", "contrapilastra", "piastra", "pilastra", "top\u00e1zio", "bacelo",
      "barrelo", "bicarelo", "biquelo", "boquelo", "botelo", "bouquelo", "cacarelo", "cachelo",
      "cadrelo", "campelo", "candelo", "cantelo", "carabelo", "carambelo", "caramelo", "cercelo",
      "cerebelo", "chocarelo", "coitelo", "conchelo", "corbelo", "cotobelo", "couselo", "destelo",
      "desvelo", "esf\u00e1celo", "fandelo", "fardelo", "farelo", "farnelo", "flabelo", "ganchelo",
      "garfelo", "involucelo", "mantelo", "montelo", "outerelo", "padicelo", "pesadelo", "pinguelo",
      "piquelo", "rampelo", "rastrelo", "restelo", "tornecelo", "trabelo", "restrelo", "portelo",
      "ourelo", "zarapelo", "arqueta", "atleta", "avoceta", "baioneta", "baldeta", "banqueta",
      "barraganeta", "barreta", "borleta", "buceta", "caceta", "calceta", "caldeta", "cambeta",
      "canaleta", "caneta", "carreta", "cerceta", "chaparreta", "chapeta", "chareta", "chincheta",
      "colcheta", "cometa", "corbeta", "corveta", "cuneta", "desteta", "espeta", "espoleta",
      "estafeta", "esteta", "faceta", "falanxeta", "frasqueta", "gaceta", "gabeta", "galleta",
      "garabeta", "gaveta", "glorieta", "lagareta", "lambeta", "lanceta", "libreta", "maceta",
      "macheta", "maleta", "malleta", "mareta", "marreta", "meseta", "mofeta", "muleta",
      "peseta", "planeta", "raqueta", "regreta", "saqueta", "veleta", "vendeta", "vi\u00f1eta",
      "alfinete", "ariete", "bacinete", "banquete", "barallete", "barrete", "billete", "binguelete",
      "birrete", "bonete", "bosquete", "bufete", "burlete", "cabalete", "cacahuete", "cavinete",
      "capacete", "carrete", "casarete", "casete", "chupete", "clarinete", "colchete", "colete",
      "capete", "curupete", "disquete", "estilete", "falsete", "ferrete", "filete", "gallardete",
      "gobelete", "inglete", "machete", "miquelete", "molete", "mosquete", "piquete", "ribete",
      "rodete", "rolete", "roquete", "sorvete", "vedete", "vendete", "andarica", "bot\u00e1nica",
      "botica", "dial\u00e9ctica", "din\u00e1mica", "f\u00edsica", "formica", "gr\u00e1fica", "marica", "t\u00fanica",
      "conico", "acetifico", "acidifico", "arpexo", "arquexo", "asexo", "axexo", "azulexo",
      "badexo", "bafexo", "bocexo", "bosquexo", "boubexo", "cacarexo", "carrexo", "cascarexo",
      "castrexo", "convexo", "cotexo", "desexo", "despexo", "forcexo", "gabexo", "gargarexo",
      "gorgolexo", "inconexo", "manexo", "merexo", "narnexo", "padexo", "patexo", "sopexo",
      "varexo", "airexa", "bandexa", "carrexa", "envexa", "igrexa", "larexa", "patexa",
      "presexa", "sobexa", "cami\u00f1o", "cari\u00f1o", "comi\u00f1o", "golfi\u00f1o", "padri\u00f1o", "sobri\u00f1o",
      "vici\u00f1o", "veci\u00f1o", "camari\u00f1a", "campi\u00f1a", "entreli\u00f1a", "espi\u00f1a", "fari\u00f1a", "mori\u00f1a",
      "vali\u00f1a", "anaroide", "aneroide", "asteroide", "axoide", "cardioide", "celuloide", "coronoide",
      "discoide", "espermatozoide", "espiroide", "esquizoide", "esteroide", "glenoide", "linfoide", "hemorroide",
      "melaloide", "sacaroide", "tetraploide", "varioloide", "aixola", "ampola", "argola", "arola",
      "arter\u00edola", "bandola", "b\u00edtola", "bract\u00e9ola", "cachola", "carambola", "carapola", "carola",
      "carrandiola", "catrapola", "cebola", "centola", "champola", "chatola", "cirola", "c\u00edtola",
      "consola", "corola", "empola", "escarola", "esmola", "estola", "fitola", "flor\u00edcola",
      "gara\u00f1ola", "g\u00e1rgola", "garxola", "glicocola", "g\u00f3ndola", "mariola", "marola", "michola",
      "pirola", "rebola", "rup\u00edcola", "sax\u00edcola", "s\u00e9mola", "tachola", "t\u00f3mbola", "arrolo",
      "babiolo", "cacharolo", "caixarolo", "carolo", "carramolo", "cascarolo", "cirolo", "codrolo",
      "correolo", "cotrolo", "desconsolo", "rebolo", "repolo", "subsolo", "tixolo", "t\u00f3mbolo",
      "torolo", "tr\u00e9molo", "vac\u00faolo", "xermolo", "z\u00f3colo", "aigote", "alcaiote", "barbarote",
      "balote", "billote", "cachote", "camarote", "capote", "cebote", "chichote", "citote",
      "cocorote", "escote", "ga\u00f1ote", "garrote", "gavote", "lamote", "lapote", "larapote",
      "lingote", "l\u00edtote", "magote", "marrote", "matalote", "pandote", "paparote", "rebote",
      "tagarote", "zarrote", "as\u00edntota", "caiota", "cambota", "chacota", "compota", "creosota",
      "curota", "derrota", "d\u00edspota", "gamota", "maniota", "pelota", "picota", "pillota",
      "pixota", "queirota", "remota", "abrocho", "arrocho", "carocho", "falucho", "bombacho",
      "borracho", "mostacho", "borracha", "carracha", "estacha", "garnacha", "limacha", "remolacha",
      "abrocha", "caduco", "estuco", "fachuco", "malluco", "saluco", "trabuco", "carri\u00f1ouzo",
      "fachuzo", "ma\u00f1uzo", "mestruzo", "tapuzo", "barruza", "chamuza", "chapuza", "charamuza",
      "conduza", "deduza", "desluza", "entreluza", "induza", "reluza", "seduza", "traduza",
      "trasluza", "caramuxa", "carrabouxa", "cartuxa", "coruxa", "curuxa", "gaturuxa", "maruxa",
      "meruxa", "miruxa", "moruxa", "muruxa", "papuxa", "rabuxa", "trouxa", "caramuxo",
      "carouxo", "carrabouxo", "curuxo", "debuxo", "ganduxo", "influxo", "negouxo", "pertuxo",
      "refluxo", "alborello", "artello", "botello", "cachafello", "calello", "casarello", "cazabello",
      "cercello", "cocerello", "concello", "consello", "desparello", "escaravello", "espello", "fedello",
      "fervello", "gagafello", "gorrobello", "nortello", "pendello", "troupello", "trebello", "alborella",
      "bertorella", "bocatella", "botella", "calella", "cercella", "gadella", "grosella", "lentella",
      "movella", "nocella", "noitevella", "parella", "pelella", "percebella", "segorella", "sabella",
      "acridade", "calidade", "agoireiro", "bardalleiro", "braseiro", "barreiro", "canteiro", "capoeiro",
      "carneiro", "carteiro", "cinceiro", "faroleiro", "mareiro", "preguiceiro", "quinteiro", "raposeiro",
      "retranqueiro", "regueiro", "sineiro", "troleiro", "ventureiro", "cabeleira", "canteira", "cocheira",
      "folleira", "milleira", "armario", "calcario", "lionario", "salario", "cetaria", "coronaria",
      "fumaria", "linaria", "lunaria", "parietaria", "saponaria", "serpentaria", "bal\u00edstico", "ensa\u00edstico",
      "batista", "ciclista", "fadista", "operista", "tenista", "verista", "grado", "agrado",
      "agnato", "c\u00e1ndido", "c\u00e2ndido", "consolido", "decidido", "duvido", "marido", "r\u00e1pido",
      "bastida", "d\u00fabida", "dubida", "duvida", "ermida", "\u00e9xida", "guarida", "lapicida",
      "medida", "morida", "estudo", "escudo", "abada", "alhada", "allada", "pitada",
      "cambadela", "cavadela", "forcadela", "erisipidela", "mortadela", "espadela", "fondedela", "picadela",
      "arandela", "candela", "cordela", "escudela", "pardela", "canela", "capela", "cotela",
      "cubela", "curupela", "escarapela", "esparrela", "estela", "fardela", "flanela", "fornela",
      "franela", "gabela", "gamela", "gavela", "glumela", "granicela", "lamela", "lapela",
      "malvela", "manela", "manganela", "mexarela", "micela", "mistela", "novela", "ourela",
      "panela", "parcela", "pasarela", "patamela", "patela", "paxarela", "pipela", "pitela",
      "postela", "pubela", "restela", "sabela", "salmonela", "secuela", "sentinela", "soldanela",
      "subela", "temoncela", "tesela", "tixela", "tramela", "trapela", "varela", "vitela",
      "xanela", "xestela", "af\u00e1bel", "fi\u00e1bel", "cr\u00edbel", "impos\u00edbel", "pos\u00edbel", "fis\u00edbel",
      "fal\u00edbel", "alimente", "adiante", "acrescente", "elefante", "frequente", "freq\u00fcente", "gigante",
      "instante", "oriente", "permanente", "posante", "possante", "restaurante", "acracia", "audacia",
      "falacia", "farmacia", "caricia", "delicia", "ledicia", "malicia", "milicia", "noticia",
      "pericia", "presbicia", "primicia", "regalicia", "sevicia", "tiricia", "alvariza", "baliza",
      "cachiza", "caniza", "ca\u00f1iza", "carbaliza", "carriza", "chamariza", "chapiza", "fraguiza",
      "latiza", "longaniza", "ma\u00f1iza", "nabiza", "peliza", "preguiza", "rabiza", "palmexar",
      "aeraci\u00f3n", "condici\u00f3n", "gornici\u00f3n", "monici\u00f3n", "nutrici\u00f3n", "petici\u00f3n", "posici\u00f3n", "sedici\u00f3n",
      "volici\u00f3n", "abrasi\u00f3n", "alusi\u00f3n", "armaz\u00f3n", "abal\u00f3n", "acorde\u00f3n", "alci\u00f3n", "aldrab\u00f3n",
      "aler\u00f3n", "ali\u00f1\u00f3n", "amb\u00f3n", "bomb\u00f3n", "calz\u00f3n", "camp\u00f3n", "canal\u00f3n", "cant\u00f3n",
      "capit\u00f3n", "ca\u00f1\u00f3n", "cent\u00f3n", "cicl\u00f3n", "coll\u00f3n", "colof\u00f3n", "cop\u00f3n", "cot\u00f3n",
      "cup\u00f3n", "pet\u00f3n", "tir\u00f3n", "tour\u00f3n", "tur\u00f3n", "unci\u00f3n", "versi\u00f3n", "zub\u00f3n",
      "zurr\u00f3n", "abandona", "acetona", "aleurona", "amazona", "an\u00e9mona", "bombona", "cambona",
      "carona", "chacona", "charamona", "cincona", "condona", "cortisona", "cretona", "detona",
      "estona", "fitohormona", "fregona", "gerona", "hidroquinona", "hormona", "lesiona", "madona",
      "maratona", "matrona", "metadona", "mon\u00f3tona", "neurona", "pamplona", "peptona", "poltrona",
      "proxesterona", "quinona", "silicona", "sulfona", "abandoa", "madroa", "barbacoa", "estoa",
      "airoa", "eiroa", "amalloa", "\u00e1mboa", "am\u00e9ndoa", "anchoa", "antin\u00e9boa", "av\u00e9ntoa",
      "avoa", "b\u00e1goa", "balboa", "bisavoa", "boroa", "canoa", "caroa", "comadroa",
      "coroa", "\u00e9ngoa", "esp\u00e1coa", "filloa", "f\u00edrgoa", "gra\u00f1oa", "lagoa", "lanzoa",
      "magoa", "m\u00e1moa", "morzoa", "noiteboa", "noraboa", "para\u00f1oa", "persoa", "queiroa",
      "ra\u00f1oa", "t\u00e1boa", "tataravoa", "teiroa", "alpaca", "barraca", "bullaca", "buraca",
      "carraca", "casaca", "cavaca", "cloaca", "entresaca", "ervellaca", "espinaca", "estaca",
      "farraca", "millaca", "pastinaca", "pataca", "resaca", "urraca", "purraca", "afinal",
      "animal", "estatal", "bisexual", "bissexual", "desleal", "fiscal", "formal", "pessoal",
      "persoal", "liberal", "postal", "virtual", "visual", "pontual", "puntual", "homosexual",
      "heterosexual", "abaixador", "autor", "motor", "pastor", "pintor", "asesor", "assessor",
      "favor", "mellor", "melhor", "redor", "rigor", "sensor", "tambor", "tumor",
      "albacora", "an\u00e1fora", "\u00e1ncora", "apisoadora", "ardora", "ascospora", "aurora", "av\u00e9spora",
      "bit\u00e1cora", "can\u00e9fora", "cantimplora", "cat\u00e1fora", "cepilladora", "demora", "descalcificadora", "di\u00e1spora",
      "empacadora", "ep\u00edfora", "ecavadora", "escora", "eslora", "espora", "fotocompo\u00f1edora", "fotocopiadora",
      "grampadora", "is\u00edcora", "lavadora", "lixadora", "macrospora", "madr\u00e9pora", "madr\u00e1gora", "masora",
      "mellora", "met\u00e1fora", "microspora", "mil\u00e9pora", "milp\u00e9ndora", "n\u00e9cora", "oospora", "padeadora",
      "pasiflora", "p\u00e9cora", "p\u00edldora", "p\u00f3lvora", "ratinadora", "r\u00e9mora", "retroescavadora", "s\u00f3fora",
      "torradora", "tr\u00e9mbora", "uredospora", "v\u00edbora", "v\u00edncora", "zoospora", "librar\u00eda", "aluaxe",
      "amaraxe", "amperaxe", "bagaxe", "balaxe", "barcaxe", "borraxe", "bescaxe", "cabotaxe",
      "carraxe", "cartilaxe", "chantaxe", "colaxe", "coraxe", "carruaxe", "dragaxe", "embalaxe",
      "ensilaxe", "epistaxe", "fagundaxe", "fichaxe", "fogaxe", "forraxe", "fretaxe", "friaxe",
      "garaxe", "homenaxe", "leitaxe", "li\u00f1axe", "listaxe", "maraxe", "marcaxe", "maridaxe",
      "masaxe", "miraxe", "montaxe", "pasaxe", "peaxe", "portaxe", "ramaxe", "rebelaxe",
      "rodaxe", "romaxe", "sintaxe", "sondaxe", "tiraxe", "vantaxe", "vendaxe", "viraxe",
      "alteza", "beleza", "fereza", "fineza", "vasteza", "vileza", "acidez", "adultez",
      "adustez", "avidez", "candidez", "mudez", "nenez", "nudez", "pomez", "corego",
      "derrego", "entrego", "lamego", "sarego", "sartego", "afanoso", "algoso", "caldoso",
      "caloso", "cocoso", "ditoso", "favoso", "fogoso", "lamoso", "mecoso", "mocoso",
      "precioso", "rixoso", "venoso", "viroso", "xesoso", "mucosa", "glicosa", "baldosa",
      "celulosa", "isoglosa", "nitrocelulosa", "levulosa", "ortosa", "pectosa", "preciosa", "sacarosa",
      "serosa", "ventosa", "agrume", "albume", "alcume", "batume", "cacume", "cerrume",
      "chorume", "churume", "costume", "curtume", "estrume", "gafume", "legume", "perfume",
      "queixume", "zarrume", "albura", "armadura", "imatura", "costura", "abril", "alfil",
      "anil", "atril", "badil", "baril", "barril", "brasil", "cadril", "candil",
      "cantil", "carril", "chamil", "chancil", "civil", "cubil", "d\u00e1til", "dif\u00edcil",
      "d\u00f3cil", "edil", "est\u00e9ril", "f\u00e1cil", "fr\u00e1xil", "funil", "fusil", "gr\u00e1cil",
      "gradil", "h\u00e1bil", "hostil", "marfil", "pasivo", "positivo", "passivo", "possessivo",
      "posesivo", "pexotarivo", "relativo", "azar", "bazar", "patamar", "faroeste", "agreste",
      "enfian", "eloxian", "ensaian", "admirei", "ampliava", "arara", "prepara", "alvar\u00e1",
      "bacar\u00e1", "prepare", "agrava", "acelera", "espera", "espere", "reprimo", "intimo",
      "\u00edntimo", "nimo", "queimo", "ximo", "fronteira", "s\u00e1tira", "alfabetizar", "organizar",
      "acreditar", "explicitar", "estreitar", "adquire", "alardear", "nuclear", "\u00e9ter", "pier",
      "chapeu", "est\u00f3ria", "fatia", "acia", "praia", "elogia", "mania", "l\u00e1bia",
      "aprecia", "pol\u00edcia", "arredia", "cheia", "\u00e1sia", "rede", "b\u00edpede", "c\u00e9spede",
      "parede", "palm\u00edpede", "vostede", "h\u00f3spede", "adrede", "ondo", "azougue", "dengue",
      "merengue", "nurague", "rengue", "alambique", "albaricoque", "abaroque", "alcrique", "almadraque",
      "almanaque", "arenque", "arinque", "baduloque", "ballestrinque", "betoque", "bivaque", "bloque",
      "bodaque", "bosque", "breque", "buque", "cacique", "cheque", "claque", "contradique",
      "coque", "croque", "dique", "duque", "enroque", "espeque", "estoque", "estoraque",
      "estraloque", "estrinque", "milicroque", "monicreque", "orinque", "palenque", "parque", "penique",
      "picabeque", "pique", "psique", "raque", "remolque", "xeque", "repenique", "roque",
      "sotobosque", "tabique", "tanque", "toque", "traque", "truque", "vivaque", "xaque",
      "amasadela", "cerva", "marte", "barro", "fado", "cabo", "libro", "cervo",
      "amanh\u00e3", "arapu\u00e3", "f\u00e3", "div\u00e3", "manh\u00e3", "i",
    });
  }
}
