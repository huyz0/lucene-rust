import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import org.apache.lucene.search.spell.JaroWinklerDistance;
import org.apache.lucene.search.spell.LevenshteinDistance;
import org.apache.lucene.search.spell.LuceneLevenshteinDistance;
import org.apache.lucene.search.spell.NGramDistance;
import org.apache.lucene.search.spell.PlainTextDictionary;
import org.apache.lucene.search.spell.StringDistance;
import org.apache.lucene.search.spell.SuggestWord;
import org.apache.lucene.search.spell.SuggestWordFrequencyComparator;
import org.apache.lucene.search.spell.SuggestWordQueue;
import org.apache.lucene.search.spell.SuggestWordScoreComparator;
import org.apache.lucene.search.suggest.InputIterator;
import org.apache.lucene.util.BytesRef;

/**
 * M13 T13.3e: the spell package's string distances, {@code SuggestWord} comparators, {@code
 * SuggestWordQueue} and {@code PlainTextDictionary}. Writes {@code spell_distances/}: {@code
 * words.tsv} (one word per line, the line number is its index), {@code distances.tsv} (distance,
 * i, j, {@code Float.floatToIntBits} in hex), {@code suggest_words.tsv} (string, score bits, freq,
 * {@code toString()}), {@code orders.tsv} (comparator, the sorted indexes), {@code queues.tsv}
 * (comparator, size, the popped indexes) and {@code plain_text.tsv} (input name, the entries).
 */
public class GenSpellDistances {
  static final String[] WORDS = {
    "",
    "a",
    "b",
    "ab",
    "ba",
    "abc",
    "acb",
    "bca",
    "abcd",
    "abdc",
    "badc",
    "lucene",
    "lucen",
    "lucnee",
    "ulcene",
    "lucenes",
    "luxene",
    "solr",
    "sloar",
    "martha",
    "marhta",
    "dwayne",
    "duane",
    "dixon",
    "dicksonx",
    "jones",
    "johnson",
    "café",
    "cafe",
    "caffè",
    "naïve",
    "naive",
    "日本語",
    "日本",
    "😀a",
    "a😀",
    "😀😃",
    "😃😀",
    "aaaa",
    "aaab",
    "abababab",
    "babababa",
    "spellchecker",
    "spelchecker",
    "spellcheker",
    "x",
  };

  static String bits(float f) {
    return Integer.toHexString(Float.floatToIntBits(f));
  }

  record Word(String string, float score, int freq) {
    SuggestWord toSuggestWord() {
      SuggestWord w = new SuggestWord();
      w.string = string;
      w.score = score;
      w.freq = freq;
      return w;
    }
  }

  static final Word[] SUGGEST_WORDS = {
    new Word("apple", 0.5f, 10),
    new Word("apply", 0.5f, 10),
    new Word("ample", 0.5f, 3),
    new Word("maple", 0.75f, 3),
    new Word("appel", 0.75f, 10),
    new Word("apples", 0.25f, 100),
    new Word("aple", 0.25f, 100),
    new Word("Apple", 0.5f, 10),
    new Word("äpple", 0.5f, 10),
    new Word("😀apple", 0.5f, 10),
    new Word("Ａpple", 0.5f, 10),
    new Word("nan", Float.NaN, 5),
    new Word("nan2", Float.NaN, 5),
    new Word("negzero", -0.0f, 5),
    new Word("poszero", 0.0f, 5),
    new Word("neg", -1.5f, -2),
    new Word("big", 1e-7f, Integer.MAX_VALUE),
    new Word("apple", 0.5f, 10),
  };

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("spell_distances");
    Files.createDirectories(out);

    StringBuilder words = new StringBuilder();
    for (String w : WORDS) words.append(w).append('\n');
    Files.writeString(out.resolve("words.tsv"), words.toString(), StandardCharsets.UTF_8);

    List<Object[]> distances = new ArrayList<>();
    distances.add(new Object[] {"levenshtein", new LevenshteinDistance()});
    distances.add(new Object[] {"lucene_levenshtein", new LuceneLevenshteinDistance()});
    distances.add(new Object[] {"jarowinkler", new JaroWinklerDistance()});
    JaroWinklerDistance jw0 = new JaroWinklerDistance();
    jw0.setThreshold(0f);
    distances.add(new Object[] {"jarowinkler_0", jw0});
    JaroWinklerDistance jw1 = new JaroWinklerDistance();
    jw1.setThreshold(1.1f);
    distances.add(new Object[] {"jarowinkler_1.1", jw1});
    for (int n : new int[] {0, 1, 2, 3, 4}) {
      distances.add(new Object[] {"ngram_" + n, new NGramDistance(n)});
    }
    StringBuilder rows = new StringBuilder();
    for (Object[] d : distances) {
      StringDistance sd = (StringDistance) d[1];
      for (int i = 0; i < WORDS.length; i++) {
        for (int j = 0; j < WORDS.length; j++) {
          rows.append(d[0])
              .append('\t')
              .append(i)
              .append('\t')
              .append(j)
              .append('\t')
              .append(bits(sd.getDistance(WORDS[i], WORDS[j])))
              .append('\n');
        }
      }
    }
    Files.writeString(out.resolve("distances.tsv"), rows.toString(), StandardCharsets.UTF_8);

    StringBuilder sw = new StringBuilder();
    for (Word w : SUGGEST_WORDS) {
      sw.append(w.string)
          .append('\t')
          .append(bits(w.score))
          .append('\t')
          .append(w.freq)
          .append('\t')
          .append(w.toSuggestWord())
          .append('\n');
    }
    Files.writeString(out.resolve("suggest_words.tsv"), sw.toString(), StandardCharsets.UTF_8);

    List<Object[]> comparators = new ArrayList<>();
    comparators.add(new Object[] {"score", new SuggestWordScoreComparator()});
    comparators.add(new Object[] {"frequency", new SuggestWordFrequencyComparator()});
    StringBuilder orders = new StringBuilder();
    StringBuilder queues = new StringBuilder();
    for (Object[] c : comparators) {
      @SuppressWarnings("unchecked")
      Comparator<SuggestWord> cmp = (Comparator<SuggestWord>) c[1];
      for (boolean reverse : new boolean[] {false, true}) {
        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < SUGGEST_WORDS.length; i++) idx.add(i);
        Comparator<SuggestWord> eff = reverse ? Collections.reverseOrder(cmp) : cmp;
        idx.sort((x, y) -> eff.compare(SUGGEST_WORDS[x].toSuggestWord(), SUGGEST_WORDS[y].toSuggestWord()));
        orders.append(c[0]).append(reverse ? "_reverse" : "").append('\t').append(join(idx)).append('\n');
      }
      for (int size : new int[] {1, 2, 3, 5, 8, SUGGEST_WORDS.length, SUGGEST_WORDS.length + 3}) {
        SuggestWordQueue q =
            c[0].equals("score") && size % 2 == 1
                ? new SuggestWordQueue(size)
                : new SuggestWordQueue(size, cmp);
        List<SuggestWord> made = new ArrayList<>();
        for (Word w : SUGGEST_WORDS) {
          SuggestWord s = w.toSuggestWord();
          made.add(s);
          q.insertWithOverflow(s);
        }
        List<Integer> popped = new ArrayList<>();
        while (q.size() > 0) {
          SuggestWord s = q.pop();
          for (int i = 0; i < made.size(); i++) {
            if (made.get(i) == s) popped.add(i);
          }
        }
        queues.append(c[0]).append('\t').append(size).append('\t').append(join(popped)).append('\n');
      }
    }
    Files.writeString(out.resolve("orders.tsv"), orders.toString(), StandardCharsets.UTF_8);
    Files.writeString(out.resolve("queues.tsv"), queues.toString(), StandardCharsets.UTF_8);

    String[][] texts = {
      {"empty", ""},
      {"one", "word"},
      {"lf", "alpha\nbeta\ngamma\n"},
      {"crlf", "alpha\r\nbeta\r\n\r\ngamma"},
      {"cr", "alpha\rbeta\r\rgamma\r"},
      {"blank_lines", "\n\n\nx\n\n"},
      {"unicode", "café\n日本語\n😀\n"},
      {"tabs", "a\tb\tc\nd e\n"},
    };
    StringBuilder plain = new StringBuilder();
    for (String[] t : texts) {
      PlainTextDictionary dict = new PlainTextDictionary(new StringReader(t[1]));
      InputIterator it = dict.getEntryIterator();
      plain.append(t[0]).append('\t').append(it.hasPayloads()).append('\t').append(it.hasContexts());
      BytesRef b;
      while ((b = it.next()) != null) {
        plain.append('\t').append(hex(b)).append(':').append(it.weight());
      }
      plain.append('\n');
    }
    Files.writeString(out.resolve("plain_text.tsv"), plain.toString(), StandardCharsets.UTF_8);
  }

  static String hex(BytesRef b) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < b.length; i++) {
      sb.append(String.format("%02x", b.bytes[b.offset + i] & 0xff));
    }
    return sb.toString();
  }

  static String join(List<Integer> xs) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < xs.size(); i++) {
      if (i > 0) sb.append(',');
      sb.append(xs.get(i));
    }
    return sb.toString();
  }
}
