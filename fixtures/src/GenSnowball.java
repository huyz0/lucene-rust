import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.KeywordTokenizer;
import org.apache.lucene.analysis.snowball.SnowballFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.tartarus.snowball.Among;
import org.tartarus.snowball.SnowballStemmer;

/**
 * Snowball stems for every stemmer Lucene 10.5.0 ships, through Lucene's own {@code
 * SnowballFilter} (one {@code KeywordTokenizer} token per word), for {@code
 * crates/lucene-analysis/tests/snowball_fixtures.rs}.
 *
 * <p>The words are this generator's own, so no third-party vocabulary is redistributed: for each
 * language, every string of every {@code Among} table of its {@code
 * org.tartarus.snowball.ext.*Stemmer} (read by reflection -- the suffixes, prefixes and exception
 * words the algorithm branches on) appended to and prefixed with random stems over the language's
 * alphabet (the characters of those strings), random stem + suffix chains (fixed seed), and the
 * lowercased words of {@code fixtures/corpus/analysis-common.txt}. A handful of words carry a
 * character outside the Basic Multilingual Plane, where Java's UTF-16 stemmers and the port's
 * UTF-8 ones may disagree: they go to {@code supplementary.words} instead, as the record of what
 * Java does there.
 *
 * <p>Output: {@code snowball/<Language>.words}, one line per word, {@code word\tstem}, or just
 * {@code word} when the stem is the word itself. Deterministic.
 */
public class GenSnowball {
  static final String[] LANGUAGES = {
    "Arabic", "Armenian", "Basque", "Catalan", "Danish", "Dutch", "English", "Estonian", "Finnish",
    "French", "German", "Greek", "Hindi", "Hungarian", "Indonesian", "Irish", "Italian",
    "Lithuanian", "Nepali", "Norwegian", "Porter", "Portuguese", "Romanian", "Russian", "Serbian",
    "Spanish", "Swedish", "Tamil", "Turkish", "Yiddish"
  };

  static final int CANDIDATES = 150000;
  static final int GROWTH_CALLS = 400000;

  static final String[] SUPPLEMENTARY = {
    "😀", "running😀", "😀ing", "a😀b", "𝒜ction", "lov𝒜es", "😀😀s", "𐍈ations"
  };

  static String stem(Analyzer a, String w) throws IOException {
    try (TokenStream ts = a.tokenStream("f", w)) {
      CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
      ts.reset();
      if (!ts.incrementToken()) throw new IllegalStateException("no token for " + w);
      String s = term.toString();
      if (ts.incrementToken()) throw new IllegalStateException("two tokens for " + w);
      ts.end();
      return s;
    }
  }

  static List<String> amongStrings(String lang) throws Exception {
    Class<?> c = Class.forName("org.tartarus.snowball.ext." + lang + "Stemmer");
    Class<?> among = Class.forName("org.tartarus.snowball.Among");
    Field sField = among.getDeclaredField("s");
    sField.setAccessible(true);
    List<String> out = new ArrayList<>();
    for (Field f : c.getDeclaredFields()) {
      if (Modifier.isStatic(f.getModifiers()) && f.getType().isArray()
          && f.getType().getComponentType() == among) {
        f.setAccessible(true);
        for (Object a : (Object[]) f.get(null)) {
          String s = new String((char[]) sField.get(a));
          if (!s.isEmpty()) out.add(s);
        }
      }
    }
    return out;
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("snowball");
    Files.createDirectories(out);
    String corpusDir = System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus");
    String corpus =
        Files.readString(Path.of(corpusDir, "analysis-common.txt"), StandardCharsets.UTF_8);
    Set<String> corpusWords = new TreeSet<>();
    for (String w : corpus.toLowerCase(Locale.ROOT).split("[^\\p{L}\\p{M}\\p{Nd}'’]+")) {
      if (!w.isEmpty() && w.codePointCount(0, w.length()) == w.length()) corpusWords.add(w);
    }
    StringBuilder supp = new StringBuilder();
    for (String lang : LANGUAGES) {
      List<String> amongs = amongStrings(lang);
      // Two alphabets: the characters of the among strings (first-seen order),
      // and those widened to every lower-case or uncased letter or mark of the
      // 128-character blocks they fall in (a-z and Latin-1 for Latin script),
      // so groupings and normalisations see characters no suffix holds.
      Set<Character> alpha = new LinkedHashSet<>();
      for (String s : amongs) for (char ch : s.toCharArray()) alpha.add(ch);
      Set<Character> wide = new LinkedHashSet<>(alpha);
      for (char ch : alpha) {
        int lo = ch < 0x80 ? 0x61 : ch & ~0x7F;
        int hi = ch < 0x80 ? 0xFF : ch | 0x7F;
        for (int c = lo; c <= hi; c++) {
          int type = Character.getType(c);
          boolean letter = Character.isLetter(c) && !Character.isUpperCase(c);
          boolean mark =
              type == Character.NON_SPACING_MARK || type == Character.COMBINING_SPACING_MARK;
          if ((letter || mark) && c != 0xAA && c != 0xBA) wide.add((char) c);
        }
      }
      char[] letters = chars(alpha);
      char[] wideLetters = chars(wide);
      Random rnd = new Random(lang.hashCode());
      Set<String> words = new LinkedHashSet<>();
      for (String s : amongs) {
        words.add(s);
        words.add(base(rnd, letters, wideLetters) + s);
        words.add(base(rnd, letters, wideLetters) + base(rnd, letters, wideLetters) + s);
        words.add(s + base(rnd, letters, wideLetters));
        words.add(amongs.get(rnd.nextInt(amongs.size())) + base(rnd, letters, wideLetters) + s);
        words.add(base(rnd, letters, wideLetters) + s + amongs.get(rnd.nextInt(amongs.size())));
      }
      words.addAll(corpusWords);
      // Upper case and single characters: groupings miss, regions are empty.
      for (char ch : wideLetters) words.add(String.valueOf(ch));
      words.add(amongs.get(0).toUpperCase(Locale.ROOT) + "S");
      try (Analyzer a =
          new Analyzer() {
            @Override
            protected TokenStreamComponents createComponents(String field) {
              Tokenizer t = new KeywordTokenizer();
              return new TokenStreamComponents(t, new SnowballFilter(t, lang));
            }
          }) {
        // Random prefix + stem + suffix chains: CANDIDATES of them, each kept
        // when the traced stemmer makes a decision no kept word made (an among
        // entry, a literal or a grouping test with a new outcome), and every
        // 200th regardless.
        SnowballStemmer tracer = traced(lang);
        Set<String> seen = new HashSet<>();
        T.LITERALS.clear();
        for (String w : words) trace(tracer, w, seen);
        // The pieces words are built from: the among strings, then the
        // literals the program tested (`'ki'`), which no table lists.
        List<String> pieces = new ArrayList<>(amongs);
        for (String l : T.LITERALS) if (!amongs.contains(l)) pieces.add(l);
        for (int i = 0; i < CANDIDATES; i++) {
          StringBuilder w = new StringBuilder();
          if (rnd.nextInt(6) == 0) w.append(pieces.get(rnd.nextInt(pieces.size())));
          w.append(base(rnd, letters, wideLetters));
          int n = rnd.nextInt(5);
          for (int j = 0; j < n; j++) w.append(pieces.get(rnd.nextInt(pieces.size())));
          String word = w.toString();
          if (words.contains(word)) continue;
          if (trace(tracer, word, seen) || i % 200 == 0) words.add(word);
        }
        // Then grow every word that made a new decision by one more among
        // string at either end, breadth first, keeping the growths that make
        // new decisions in turn: suffix chains several entries deep (Turkish,
        // Greek) are out of reach of random chains.
        java.util.ArrayDeque<String> frontier = new java.util.ArrayDeque<>(words);
        for (int calls = 0; calls < GROWTH_CALLS && !frontier.isEmpty(); ) {
          String w = frontier.poll();
          for (int k = 0; k < 24; k++, calls++) {
            String s = pieces.get(rnd.nextInt(pieces.size()));
            String grown = k % 4 == 3 ? s + w : w + s;
            if (!words.contains(grown) && trace(tracer, grown, seen)) {
              words.add(grown);
              frontier.add(grown);
            }
          }
        }
        StringBuilder sb = new StringBuilder();
        for (String w : words) {
          if (w.indexOf('\t') >= 0 || w.indexOf('\n') >= 0) continue;
          String s = stem(a, w);
          sb.append(w);
          if (!s.equals(w)) sb.append('\t').append(s);
          sb.append('\n');
        }
        Files.writeString(out.resolve(lang + ".words"), sb.toString(), StandardCharsets.UTF_8);
        for (String w : SUPPLEMENTARY) {
          supp.append(lang).append('\t').append(w).append('\t').append(stem(a, w)).append('\n');
        }
      }
    }
    Files.writeString(out.resolve("supplementary.words"), supp.toString(), StandardCharsets.UTF_8);
  }

  /**
   * The language's stemmer with every decision the runtime makes recorded in {@link T}: which
   * entry of which {@code among} table matched, whether a literal or a grouping test passed. The
   * tables and groupings are numbered in first-use order, so the record is deterministic.
   */
  static SnowballStemmer traced(String lang) {
    switch (lang) {
      case "Arabic":
        return new org.tartarus.snowball.ext.ArabicStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Armenian":
        return new org.tartarus.snowball.ext.ArmenianStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Basque":
        return new org.tartarus.snowball.ext.BasqueStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Catalan":
        return new org.tartarus.snowball.ext.CatalanStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Danish":
        return new org.tartarus.snowball.ext.DanishStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Dutch":
        return new org.tartarus.snowball.ext.DutchStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "English":
        return new org.tartarus.snowball.ext.EnglishStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Estonian":
        return new org.tartarus.snowball.ext.EstonianStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Finnish":
        return new org.tartarus.snowball.ext.FinnishStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "French":
        return new org.tartarus.snowball.ext.FrenchStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "German":
        return new org.tartarus.snowball.ext.GermanStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Greek":
        return new org.tartarus.snowball.ext.GreekStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Hindi":
        return new org.tartarus.snowball.ext.HindiStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Hungarian":
        return new org.tartarus.snowball.ext.HungarianStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Indonesian":
        return new org.tartarus.snowball.ext.IndonesianStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Irish":
        return new org.tartarus.snowball.ext.IrishStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Italian":
        return new org.tartarus.snowball.ext.ItalianStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Lithuanian":
        return new org.tartarus.snowball.ext.LithuanianStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Nepali":
        return new org.tartarus.snowball.ext.NepaliStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Norwegian":
        return new org.tartarus.snowball.ext.NorwegianStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Porter":
        return new org.tartarus.snowball.ext.PorterStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Portuguese":
        return new org.tartarus.snowball.ext.PortugueseStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Romanian":
        return new org.tartarus.snowball.ext.RomanianStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Russian":
        return new org.tartarus.snowball.ext.RussianStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Serbian":
        return new org.tartarus.snowball.ext.SerbianStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Spanish":
        return new org.tartarus.snowball.ext.SpanishStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Swedish":
        return new org.tartarus.snowball.ext.SwedishStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Tamil":
        return new org.tartarus.snowball.ext.TamilStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Turkish":
        return new org.tartarus.snowball.ext.TurkishStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      case "Yiddish":
        return new org.tartarus.snowball.ext.YiddishStemmer() {
          @Override protected int find_among(Among[] v) { return T.among(v, super.find_among(v)); }
          @Override protected int find_among_b(Among[] v) { return T.among(v, super.find_among_b(v)); }
          @Override protected boolean eq_s(CharSequence s) { return T.eq(s, super.eq_s(s)); }
          @Override protected boolean eq_s_b(CharSequence s) { return T.eq(s, super.eq_s_b(s)); }
          @Override protected boolean in_grouping(char[] g, int lo, int hi) { return T.g(g, 0, super.in_grouping(g, lo, hi)); }
          @Override protected boolean in_grouping_b(char[] g, int lo, int hi) { return T.g(g, 1, super.in_grouping_b(g, lo, hi)); }
          @Override protected boolean out_grouping(char[] g, int lo, int hi) { return T.g(g, 2, super.out_grouping(g, lo, hi)); }
          @Override protected boolean out_grouping_b(char[] g, int lo, int hi) { return T.g(g, 3, super.out_grouping_b(g, lo, hi)); }
        };
      default:
        throw new IllegalArgumentException(lang);
    }
  }

  /** The decisions of the current word ({@link #traced}). */
  static final class T {
    static final Map<Object, Integer> IDS = new IdentityHashMap<>();
    static final Set<String> WORD = new HashSet<>();
    static String prev = "";

    /** Records a decision with the one before it, so a table consulted in two places counts twice. */
    static void hit(String d) {
      WORD.add(prev + ">" + d);
      prev = d;
    }

    static int id(Object table) {
      return IDS.computeIfAbsent(table, k -> IDS.size());
    }

    static int among(Among[] v, int r) {
      hit("a" + id(v) + "=" + r);
      return r;
    }

    static final Set<String> LITERALS = new java.util.TreeSet<>();

    static boolean eq(CharSequence s, boolean r) {
      if (s.length() > 0) LITERALS.add(s.toString());
      hit("e" + s + "=" + r);
      return r;
    }

    static boolean g(char[] g, int kind, boolean r) {
      hit("g" + kind + "." + id(g) + "=" + r);
      return r;
    }
  }

  static String base(Random rnd, char[] letters, char[] wide) {
    int n = 1 + rnd.nextInt(7);
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) {
      char[] from = rnd.nextBoolean() ? letters : wide;
      b.append(from[rnd.nextInt(from.length)]);
    }
    return b.toString();
  }

  static char[] chars(Set<Character> set) {
    char[] out = new char[set.size()];
    int k = 0;
    for (char ch : set) out[k++] = ch;
    return out;
  }

  /** Stems {@code w} with the traced stemmer; whether it made a decision not in {@code seen}. */
  static boolean trace(SnowballStemmer stemmer, String w, Set<String> seen) {
    T.WORD.clear();
    T.prev = "";
    stemmer.setCurrent(w);
    stemmer.stem();
    return seen.addAll(T.WORD);
  }

}
