import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import org.apache.commons.codec.language.Caverphone1;
import org.apache.commons.codec.language.Caverphone2;
import org.apache.commons.codec.language.ColognePhonetic;
import org.apache.commons.codec.language.DaitchMokotoffSoundex;
import org.apache.commons.codec.language.DoubleMetaphone;
import org.apache.commons.codec.language.MatchRatingApproachEncoder;
import org.apache.commons.codec.language.Metaphone;
import org.apache.commons.codec.language.Nysiis;
import org.apache.commons.codec.language.RefinedSoundex;
import org.apache.commons.codec.language.Soundex;
import org.apache.commons.codec.language.bm.Lang;
import org.apache.commons.codec.language.bm.Languages.LanguageSet;
import org.apache.commons.codec.language.bm.NameType;
import org.apache.commons.codec.language.bm.PhoneticEngine;
import org.apache.commons.codec.language.bm.RuleType;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.util.BytesRef;

/**
 * M12 T12.3: Lucene's analysis-phonetic module and the Commons Codec 1.17.2 encoders it runs.
 *
 * <p>Writes {@code analysis_phonetic/}:
 *
 * <ul>
 *   <li>{@code words.tsv}: a word list through every non-Beider-Morse encoder and option, one
 *       column each (header {@code #word<TAB>column...}); a value is {@code =code}, {@code ~} for
 *       {@code null} or {@code !ExceptionSimpleName}, escaped as {@link AnalysisRows#esc}.
 *   <li>{@code bm.tsv}: the same for Beider-Morse ({@code PhoneticEngine} name type x rule type x
 *       concat, caller language sets, and {@code Lang.guessLanguage}).
 *   <li>{@code prefixes.txt}: each name type's {@code PhoneticEngine.NAME_PREFIXES} in the order its
 *       {@code HashSet} iterates (the first matching prefix wins).
 *   <li>{@code <config>.tsv} for every configuration of {@code corpus/analysis-phonetic.conf}, as
 *       {@code GenAnalysisFactories} writes them, over {@code corpus/analysis-phonetic.txt}.
 * </ul>
 *
 * <p>The word list is KStem's dictionary head words (every third; Lucene's own data) and names
 * built from the encoders' own rule tables: fragments of Daitch-Mokotoff's {@code dmrules.txt}
 * and Beider-Morse's rule patterns, letters with diacritics, separators and name prefixes, joined
 * at random under a fixed seed. No third-party word list is redistributed. {@code
 * crates/lucene-analysis-phonetic/tests/phonetic_fixtures.rs} compares.
 */
public class GenAnalysisPhonetic {

  interface Enc {
    String apply(String s) throws Exception;
  }

  static String val(Callable<String> c) {
    try {
      String v = c.call();
      return v == null ? "~" : "=" + AnalysisRows.esc(v);
    } catch (Exception e) {
      return "!" + e.getClass().getSimpleName();
    }
  }

  static List<String> kstemWords(int every) throws Exception {
    List<String> out = new ArrayList<>();
    int n = 0;
    for (int i = 1; i <= 8; i++) {
      java.lang.reflect.Field f =
          Class.forName("org.apache.lucene.analysis.en.KStemData" + i).getDeclaredField("data");
      f.setAccessible(true);
      for (String w : (String[]) f.get(null)) {
        if (n++ % every == 0) out.add(w);
      }
    }
    return out;
  }

  /** The quoted first fields of a Commons Codec rule resource (its patterns). */
  static void patterns(String resource, Set<String> into) throws Exception {
    try (InputStream in = DaitchMokotoffSoundex.class.getResourceAsStream(resource)) {
      if (in == null) throw new IllegalStateException(resource);
      BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
      String line;
      while ((line = r.readLine()) != null) {
        line = line.trim();
        if (!line.startsWith("\"")) continue;
        int end = line.indexOf('"', 1);
        if (end > 1) into.add(line.substring(1, end));
      }
    }
  }

  static List<String> fragments() throws Exception {
    Set<String> f = new TreeSet<>();
    patterns("/org/apache/commons/codec/language/dmrules.txt", f);
    for (String nt : new String[] {"gen", "ash", "sep"}) {
      for (String l : new String[] {"any", "english", "german", "polish", "french", "spanish", "hebrew", "russian"}) {
        InputStream probe =
            DaitchMokotoffSoundex.class.getResourceAsStream(
                "/org/apache/commons/codec/language/bm/" + nt + "_rules_" + l + ".txt");
        if (probe == null) continue;
        probe.close();
        patterns("/org/apache/commons/codec/language/bm/" + nt + "_rules_" + l + ".txt", f);
      }
    }
    List<String> out = new ArrayList<>(f);
    out.addAll(Arrays.asList(
        "a", "e", "i", "o", "u", "y", "b", "c", "d", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t",
        "w", "x", "z", "A", "E", "S", "K", "M", "Sch", "Mc", "Mac", "O'", "d'", "D'", "de la ",
        "van ", "von ", "da ", "bar ", "ben ", "al ", "el ", "-", "'", " ", "  ", "ß", "é", "è", "ü",
        "ö", "ä", "ñ", "ç", "ł", "ś", "ż", "ő", "ű", "ă", "ş", "ţ", "ț", "ę", "ą", "İ", "ı", "ÿ", "Ÿ",
        "Å", "æ", "ø", "œ", "Ç", "Ñ", "É", "Ä", "Ö", "Ü", "ǅ", "ŉ", "ΐ", "ﬀ", "1", "7", ".", ",",
        "&", "\t", "😀", "Ω", "я", "ough", "augh", "eau", "ault", "wicz", "witz", "ski", "sky", "tion",
        "sion", "cia", "tia", "gn", "kn", "wr", "ps", "pn", "mb", "dg", "gh", "ph", "th", "ch", "sh",
        "zh", "kh", "ck", "cz", "rz", "sz", "tsch", "dzh"));
    return out;
  }

  static List<String> names(int count, long seed) throws Exception {
    List<String> frags = fragments();
    Random rnd = new Random(seed);
    List<String> out = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      int parts = 1 + rnd.nextInt(5);
      StringBuilder b = new StringBuilder();
      for (int p = 0; p < parts; p++) b.append(frags.get(rnd.nextInt(frags.size())));
      String s = b.toString();
      switch (rnd.nextInt(4)) {
        case 0 -> s = s.toUpperCase(Locale.ROOT);
        case 1 -> s = s.isEmpty() ? s : s.substring(0, 1).toUpperCase(Locale.ROOT) + s.substring(1);
        default -> {}
      }
      out.add(s);
    }
    return out;
  }

  static final List<String> EDGES =
      Arrays.asList(
          "", " ", "  a  ", "a", "A", "ß", "SS", "ǅ", "İstanbul", "ı", "123", "a1b", "😀x", "x😀",
          "--", "-a-", "'", "d'", "d'a", "de la", "de la cruz", "van", "van gogh", "o'brien",
          "\tsmith\t", "a b", "ab", "aa", "ae", "AE", "kn", "gn", "pn", "wr", "wh", "x", "xx", "mb",
          "jose", "san jacinto", "Josef", "SUGAR", "caesar", "chianti", "michael", "mcclellan",
          "Ellenberg", "zhao", "Schmidt", "Arnoff", "Gutierrez", "Thumbail", "Ghislane", "Ghiradelli");

  static Map<String, Enc> columns() {
    Map<String, Enc> c = new LinkedHashMap<>();
    Soundex sx = new Soundex();
    c.put("soundex", sx::soundex);
    c.put("soundex_simplified", Soundex.US_ENGLISH_SIMPLIFIED::soundex);
    c.put("soundex_genealogy", Soundex.US_ENGLISH_GENEALOGY::soundex);
    c.put("refined", new RefinedSoundex()::soundex);
    for (int max : new int[] {4, 1, 8, 0}) {
      Metaphone m = new Metaphone();
      m.setMaxCodeLen(max);
      c.put("metaphone" + max, m::metaphone);
      DoubleMetaphone d = new DoubleMetaphone();
      d.setMaxCodeLen(max);
      c.put("dmp" + max, s -> d.doubleMetaphone(s, false));
      c.put("dma" + max, s -> d.doubleMetaphone(s, true));
    }
    c.put("caverphone1", new Caverphone1()::encode);
    c.put("caverphone2", new Caverphone2()::encode);
    c.put("cologne", new ColognePhonetic()::encode);
    c.put("nysiis", new Nysiis()::encode);
    c.put("nysiis_loose", new Nysiis(false)::encode);
    c.put("mra", new MatchRatingApproachEncoder()::encode);
    DaitchMokotoffSoundex dms = new DaitchMokotoffSoundex();
    c.put("dms_encode", dms::encode);
    c.put("dms_soundex", dms::soundex);
    c.put("dms_nofold", new DaitchMokotoffSoundex(false)::soundex);
    return c;
  }

  static Map<String, Enc> bmColumns() {
    Map<String, Enc> c = new LinkedHashMap<>();
    for (NameType nt : NameType.values()) {
      for (RuleType rt : new RuleType[] {RuleType.APPROX, RuleType.EXACT}) {
        PhoneticEngine e = new PhoneticEngine(nt, rt, true);
        c.put(nt.getName() + "_" + rt.getName(), e::encode);
      }
      PhoneticEngine nc = new PhoneticEngine(nt, RuleType.APPROX, false);
      c.put(nt.getName() + "_approx_noconcat", nc::encode);
      Lang lang = Lang.instance(nt);
      c.put(nt.getName() + "_guess", lang::guessLanguage);
    }
    PhoneticEngine gen = new PhoneticEngine(NameType.GENERIC, RuleType.APPROX, true);
    c.put("gen_english", s -> gen.encode(s, LanguageSet.from(new java.util.HashSet<>(List.of("english")))));
    c.put("gen_german_polish", s -> gen.encode(s, LanguageSet.from(new java.util.HashSet<>(List.of("german", "polish")))));
    c.put("gen_klingon_english", s -> gen.encode(s, LanguageSet.from(new java.util.HashSet<>(List.of("klingon", "english")))));
    PhoneticEngine few = new PhoneticEngine(NameType.GENERIC, RuleType.EXACT, true, 3);
    c.put("gen_exact_max3", few::encode);
    return c;
  }

  static void table(Path file, Map<String, Enc> cols, List<String> words) throws Exception {
    StringBuilder m = new StringBuilder("#word");
    for (String k : cols.keySet()) m.append('\t').append(k);
    m.append('\n');
    for (String w : words) {
      m.append(AnalysisRows.esc(w));
      for (Enc e : cols.values()) m.append('\t').append(val(() -> e.apply(w)));
      m.append('\n');
    }
    Files.writeString(file, m.toString(), StandardCharsets.UTF_8);
  }

  /** {@code GenAnalysisFactories.stable}, and the registry list Java prints in {@code Map.of}'s per-run order cut. */
  static String stable(Throwable e) {
    String s = GenAnalysisFactories.stable(e);
    int i = s.indexOf("must be full class name or one of ");
    return i >= 0 ? s.substring(0, i + "must be full class name or one of ".length()) : s;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("analysis_phonetic");
    Files.createDirectories(out);
    String corpusDir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");

    LinkedHashSet<String> words = new LinkedHashSet<>(EDGES);
    words.addAll(kstemWords(3));
    words.addAll(names(10000, 20261007L));
    table(out.resolve("words.tsv"), columns(), new ArrayList<>(words));

    LinkedHashSet<String> bmWords = new LinkedHashSet<>(EDGES);
    List<String> k = kstemWords(40);
    bmWords.addAll(k);
    bmWords.addAll(names(1500, 7L));
    table(out.resolve("bm.tsv"), bmColumns(), new ArrayList<>(bmWords));

    StringBuilder p = new StringBuilder();
    java.lang.reflect.Field f = PhoneticEngine.class.getDeclaredField("NAME_PREFIXES");
    f.setAccessible(true);
    @SuppressWarnings("unchecked")
    Map<NameType, Set<String>> prefixes = (Map<NameType, Set<String>>) f.get(null);
    for (NameType nt : NameType.values()) {
      p.append(nt.name()).append('\t').append(String.join(",", prefixes.get(nt))).append('\n');
    }
    Files.writeString(out.resolve("prefixes.txt"), p.toString(), StandardCharsets.UTF_8);

    List<String> lines = AnalysisRows.corpus("analysis-phonetic.txt");
    List<String> configs =
        Files.readAllLines(Path.of(corpusDir, "analysis-phonetic.conf"), StandardCharsets.UTF_8);
    for (String config : configs) {
      if (config.isEmpty() || config.startsWith("#")) continue;
      String[] fields = config.split("\t", -1);
      StringBuilder m = new StringBuilder();
      CustomAnalyzer a;
      try {
        a = GenAnalysisFactories.build(GenAnalysisFactories.parse(fields), Path.of(corpusDir));
      } catch (Exception | Error e) {
        m.append("B\t").append(e.getClass().getSimpleName()).append('\t')
            .append(AnalysisRows.esc(stable(e))).append('\n');
        Files.writeString(out.resolve(fields[0] + ".tsv"), m.toString(), StandardCharsets.UTF_8);
        continue;
      }
      try (a) {
        m.append("S\t").append(a.toString().replaceAll("@[0-9a-f]+", "")).append('\n');
        m.append(AnalysisRows.rows(a, lines));
        for (int ln = 0; ln < lines.size(); ln++) {
          m.append("N\t").append(ln).append('\t');
          try {
            BytesRef n = a.normalize("f", lines.get(ln));
            m.append(AnalysisRows.hex(n));
          } catch (Exception e) {
            m.append("X:").append(e.getClass().getSimpleName());
          }
          m.append('\n');
        }
      }
      Files.writeString(out.resolve(fields[0] + ".tsv"), m.toString(), StandardCharsets.UTF_8);
    }
  }
}
