import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.KeywordTokenizer;
import org.apache.lucene.analysis.snowball.SnowballFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

/**
 * Seeded random strings through every Snowball stemmer Lucene 10.5.0 ships ({@code SnowballFilter}
 * over a {@code KeywordTokenizer}), for {@code crates/lucene-analysis/tests/snowball_fixtures.rs}.
 *
 * <p>Where {@code GenSnowball} builds plausible words, this throws arbitrary text at the
 * stemmers: per language {@link #WORDS} strings, alternately random code points (letters of the
 * scripts the stemmers handle, any BMP character, supplementary characters -- which the stemmers
 * can split into unpaired surrogates) and random chains of the stemmer's among strings with an
 * occasional emoji between them. Output: {@code snowball-fuzz/<Language>.tsv}, {@code
 * word\tstem}, the stem with any unpaired surrogate written as U+FFFD (as the port's {@code
 * String} carries it). Deterministic: a fixed seed per language.
 */
public class GenSnowballFuzz {
  static final int WORDS = 2000;

  static final String LETTERS =
      "abcdefghijklmnopqrstuvwxyzáéíóúàèìòùâêîôûäëïöüßçñœæøåčćđšžğışőűαβγδεζηθικλμνξοπρστυφχψωάέήίόύώϊϋΐΰς"
          + "абвгдежзийклмнопрстуфхцчшщъыьэюяёђјљњћџابتثجحخدذرزسشصضطظعغفقكلمنهويىةءآأؤإئ"
          + "אבגדהוזחטיכלמנסעפצקרשתךםןףץײױकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौंः्"
          + "கஙசஞடணதநபமயரலவழளறனாிீுூெேைொோௌ்աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ'’İIΆΈ";

  static String fix(String s) {
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (Character.isHighSurrogate(c)
          && i + 1 < s.length()
          && Character.isLowSurrogate(s.charAt(i + 1))) {
        b.append(c).append(s.charAt(++i));
      } else if (Character.isSurrogate(c)) {
        b.append('�');
      } else {
        b.append(c);
      }
    }
    return b.toString();
  }

  static List<String> amongStrings(String lang) throws Exception {
    Class<?> c = Class.forName("org.tartarus.snowball.ext." + lang + "Stemmer");
    Class<?> among = Class.forName("org.tartarus.snowball.Among");
    Field sf = among.getDeclaredField("s");
    sf.setAccessible(true);
    List<String> out = new ArrayList<>();
    for (Field f : c.getDeclaredFields()) {
      if (Modifier.isStatic(f.getModifiers())
          && f.getType().isArray()
          && f.getType().getComponentType() == among) {
        f.setAccessible(true);
        for (Object o : (Object[]) f.get(null)) out.add(new String((char[]) sf.get(o)));
      }
    }
    return out;
  }

  static String stem(Analyzer a, String w) throws IOException {
    try (TokenStream ts = a.tokenStream("f", w)) {
      CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
      ts.reset();
      if (!ts.incrementToken()) throw new IllegalStateException("no token for " + w);
      String s = term.toString();
      ts.end();
      return s;
    }
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("snowball-fuzz");
    Files.createDirectories(out);
    int[] letters = LETTERS.codePoints().toArray();
    for (String lang : GenSnowball.LANGUAGES) {
      List<String> am = amongStrings(lang);
      Random r = new Random(0x5B0B ^ lang.hashCode());
      StringBuilder sb = new StringBuilder();
      try (Analyzer a =
          new Analyzer() {
            @Override
            protected TokenStreamComponents createComponents(String field) {
              Tokenizer t = new KeywordTokenizer();
              return new TokenStreamComponents(t, new SnowballFilter(t, lang));
            }
          }) {
        for (int k = 0; k < WORDS; k++) {
          StringBuilder w = new StringBuilder();
          if (k % 2 == 0) {
            int len = 1 + r.nextInt(k % 50 == 0 ? 60 : 16);
            for (int i = 0; i < len; i++) {
              int p = r.nextInt(20);
              int cp =
                  p == 0
                      ? 0x10000 + r.nextInt(0x1000)
                      : p == 1 ? 0x20 + r.nextInt(0xD7E0) : letters[r.nextInt(letters.length)];
              if (cp == '\t' || cp == '\n' || cp == '\r') cp = 'y';
              w.appendCodePoint(cp);
            }
          } else {
            int m = 1 + r.nextInt(5);
            for (int i = 0; i < m; i++) {
              if (r.nextInt(6) == 0) {
                w.appendCodePoint(r.nextBoolean() ? 0x1F600 : letters[r.nextInt(letters.length)]);
              }
              w.append(am.get(r.nextInt(am.size())));
            }
          }
          String word = w.toString();
          if (word.isEmpty()) continue;
          sb.append(word).append('\t').append(fix(stem(a, word))).append('\n');
        }
      }
      Files.writeString(out.resolve(lang + ".tsv"), sb.toString(), StandardCharsets.UTF_8);
    }
  }
}
