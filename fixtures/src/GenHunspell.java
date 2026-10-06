import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Stream;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.KeywordTokenizer;
import org.apache.lucene.analysis.hunspell.AffixedWord;
import org.apache.lucene.analysis.hunspell.DictEntries;
import org.apache.lucene.analysis.hunspell.Dictionary;
import org.apache.lucene.analysis.hunspell.Hunspell;
import org.apache.lucene.analysis.hunspell.HunspellStemFilter;
import org.apache.lucene.analysis.hunspell.NGramFragmentChecker;
import org.apache.lucene.analysis.hunspell.SortingStrategy;
import org.apache.lucene.analysis.hunspell.Suggester;
import org.apache.lucene.analysis.hunspell.TimeoutPolicy;
import org.apache.lucene.analysis.hunspell.WordFormGenerator;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

/**
 * Hunspell over this repository's own dictionaries ({@code fixtures/corpus/hunspell/<name>.aff},
 * {@code .dic} and {@code .words}, written for the port -- Lucene's test dictionaries are not
 * redistributed), for {@code crates/lucene-analysis/tests/hunspell_fixtures.rs}.
 *
 * <p>Each dictionary is loaded twice (case-sensitive, and {@code ignoreCase}) through {@code
 * SortingStrategy.inMemory()}. The words are the {@code .words} file, every {@code .dic} root, their
 * upper- and title-case forms, and two misspellings of each listed word (the middle unit dropped,
 * the first two swapped). Per word, one line:
 *
 * <pre>W  word  spell  stems  uniqueStems  longestOnly  roots  analyses  suggestions  tuned  fragment</pre>
 *
 * with lists joined by {@code |}: stems through {@code HunspellStemFilter} (dedup off, dedup on,
 * longest only), {@code Hunspell.getRoots}, {@code analyzeSimpleWord}, {@code suggest} under
 * {@code TimeoutPolicy.NO_TIMEOUT}, a {@code Suggester} with {@code
 * NGramFragmentChecker.fromAllSimpleWords(2)} and {@code proceedPastRep()}, and whether {@code
 * NGramFragmentChecker.fromWords(3, roots)} finds an impossible fragment in the word (a suggester
 * that throws writes {@code !} and the exception class). Per root,
 * {@code E  root  lookupEntries  getAllWordForms}; per dictionary, {@code G  generateAllSimpleWords}.
 * A dictionary Lucene refuses writes {@code X  exception-class  message}. Deterministic.
 */
public class GenHunspell {
  static String join(List<?> list) {
    StringBuilder sb = new StringBuilder();
    for (Object o : list) {
      if (sb.length() > 0) sb.append('|');
      sb.append(o);
    }
    return sb.toString();
  }

  static List<String> stems(Dictionary d, String word, boolean dedup, boolean longestOnly)
      throws IOException {
    Analyzer a =
        new Analyzer() {
          @Override
          protected TokenStreamComponents createComponents(String field) {
            Tokenizer t = new KeywordTokenizer();
            return new TokenStreamComponents(t, new HunspellStemFilter(t, d, dedup, longestOnly));
          }
        };
    List<String> out = new ArrayList<>();
    try (TokenStream ts = a.tokenStream("f", word)) {
      CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
      ts.reset();
      while (ts.incrementToken()) out.add(term.toString());
      ts.end();
    }
    a.close();
    return out;
  }

  /** The suggestions, or {@code !} and the exception's class (a word IGNORE empties throws). */
  static String suggestions(java.util.function.Supplier<List<String>> s) {
    try {
      return join(s.get());
    } catch (RuntimeException e) {
      return "!" + e.getClass().getSimpleName();
    }
  }

  static InputStream in(byte[] b) {
    return new ByteArrayInputStream(b);
  }

  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]).resolve("hunspell");
    Files.createDirectories(out);
    Path dir = Path.of(System.getenv().getOrDefault("FIXTURES_CORPUS", "fixtures/corpus"), "hunspell");
    List<String> names;
    try (Stream<Path> s = Files.list(dir)) {
      names =
          s.map(p -> p.getFileName().toString())
              .filter(n -> n.endsWith(".aff"))
              .map(n -> n.substring(0, n.length() - 4))
              .sorted()
              .toList();
    }
    for (String name : names) {
      byte[] aff = Files.readAllBytes(dir.resolve(name + ".aff"));
      Path dicPath = dir.resolve(name + ".dic");
      byte[] dic = Files.exists(dicPath) ? Files.readAllBytes(dicPath) : new byte[0];
      for (boolean ignoreCase : new boolean[] {false, true}) {
        StringBuilder sb = new StringBuilder();
        Dictionary d;
        try {
          d = new Dictionary(in(aff), List.of(in(dic)), ignoreCase, SortingStrategy.inMemory());
        } catch (Exception e) {
          sb.append("X\t").append(e.getClass().getSimpleName()).append('\t')
              .append(String.valueOf(e.getMessage()).replace('\n', ' ')).append('\n');
          Files.writeString(out.resolve(name + (ignoreCase ? ".ic.tsv" : ".tsv")), sb.toString(),
              StandardCharsets.UTF_8);
          continue;
        }
        Hunspell h = new Hunspell(d, TimeoutPolicy.NO_TIMEOUT, () -> {});
        String charset = new String(aff, StandardCharsets.ISO_8859_1).contains("SET UTF-8")
            ? "UTF-8" : "ISO-8859-1";
        List<String> roots = new ArrayList<>();
        for (String line : new String(dic, charset).split("\n")) roots.add(line);
        if (!roots.isEmpty()) roots.remove(0);
        Set<String> words = new LinkedHashSet<>();
        Path wordsPath = dir.resolve(name + ".words");
        List<String> listed =
            Files.exists(wordsPath)
                ? List.of(new String(Files.readAllBytes(wordsPath), charset).split("\n"))
                : List.of();
        words.addAll(listed);
        List<String> rootWords = new ArrayList<>();
        for (String r : roots) {
          String w = r.split("[/\\s]", 2)[0];
          if (!w.isEmpty()) rootWords.add(w);
        }
        words.addAll(rootWords);
        for (String w : new ArrayList<>(words)) {
          words.add(w.toUpperCase(Locale.ROOT));
          if (!w.isEmpty()) {
            words.add(Character.toUpperCase(w.charAt(0)) + w.substring(1));
          }
        }
        for (String w : listed) {
          if (w.length() > 2) {
            int mid = w.length() / 2;
            words.add(w.substring(0, mid) + w.substring(mid + 1));
            words.add("" + w.charAt(1) + w.charAt(0) + w.substring(2));
          }
        }
        Suggester tuned =
            new Suggester(d)
                .withFragmentChecker(NGramFragmentChecker.fromAllSimpleWords(2, d, () -> {}))
                .proceedPastRep();
        NGramFragmentChecker fromRoots = NGramFragmentChecker.fromWords(3, rootWords);
        WordFormGenerator gen = new WordFormGenerator(d);
        for (String w : words) {
          if (w.isEmpty()) continue;
          List<String> analyses = new ArrayList<>();
          for (AffixedWord aw : h.analyzeSimpleWord(w)) analyses.add(aw.toString());
          sb.append("W\t").append(w)
              .append('\t').append(h.spell(w) ? 1 : 0)
              .append('\t').append(join(stems(d, w, false, false)))
              .append('\t').append(join(stems(d, w, true, false)))
              .append('\t').append(join(stems(d, w, true, true)))
              .append('\t').append(join(h.getRoots(w)))
              .append('\t').append(join(analyses))
              .append('\t').append(suggestions(() -> h.suggest(w)))
              .append('\t').append(suggestions(() -> tuned.suggestNoTimeout(w, () -> {})))
              .append('\t').append(fromRoots.hasImpossibleFragmentAround(w, 0, w.length()) ? 1 : 0)
              .append('\n');
        }
        for (String r : rootWords) {
          DictEntries e = d.lookupEntries(r);
          sb.append("E\t").append(r).append('\t').append(e == null ? "null" : join(e))
              .append('\t').append(join(gen.getAllWordForms(r, () -> {}))).append('\n');
        }
        List<String> all = new ArrayList<>();
        gen.generateAllSimpleWords(aw -> all.add(aw.toString()), () -> {});
        sb.append("G\t").append(join(all)).append('\n');
        Files.writeString(out.resolve(name + (ignoreCase ? ".ic.tsv" : ".tsv")), sb.toString(),
            StandardCharsets.UTF_8);
      }
    }
  }
}
