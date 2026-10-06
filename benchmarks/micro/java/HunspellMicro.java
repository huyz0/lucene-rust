import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.hunspell.Dictionary;
import org.apache.lucene.analysis.hunspell.HunspellStemFilter;
import org.apache.lucene.analysis.hunspell.SortingStrategy;

/**
 * M11 T11.4's Hunspell pair (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_hunspell.rs}): {@code WhitespaceTokenizer} + {@code
 * HunspellStemFilter} (dedup on) per dictionary of {@code fixtures/corpus/hunspell} over the words
 * of its fixture ({@code fixtures/data/hunspell/<name>.tsv}, {@code W} lines, 100 to a document).
 * Units are tokens; output is {@code name\tns_per_token\ttokens}.
 */
public class HunspellMicro {
  static final String[] DICTIONARIES = {"affixes", "compound", "features"};

  static List<String> docs(String name) throws IOException {
    List<String> words = new ArrayList<>();
    for (String line :
        Files.readAllLines(
            Path.of("fixtures/data/hunspell/" + name + ".tsv"), StandardCharsets.UTF_8)) {
      String[] f = line.split("\t", -1);
      if (f[0].equals("W") && f[1].codePoints().noneMatch(Character::isWhitespace)) {
        words.add(f[1]);
      }
    }
    List<String> docs = new ArrayList<>();
    for (int i = 0; i < words.size(); i += 100) {
      docs.add(String.join(" ", words.subList(i, Math.min(words.size(), i + 100))));
    }
    return docs;
  }

  public static void main(String[] args) throws Exception {
    for (String name : DICTIONARIES) {
      Path dir = Path.of("fixtures/corpus/hunspell");
      Dictionary d;
      try (InputStream aff = Files.newInputStream(dir.resolve(name + ".aff"));
          InputStream dic = Files.newInputStream(dir.resolve(name + ".dic"))) {
        d = new Dictionary(aff, List.of(dic), false, SortingStrategy.inMemory());
      }
      List<String> docs = docs(name);
      Analyzer a =
          new Analyzer() {
            @Override
            protected TokenStreamComponents createComponents(String field) {
              Tokenizer t = new WhitespaceTokenizer();
              return new TokenStreamComponents(t, new HunspellStemFilter(t, d, true, false));
            }
          };
      SweepMicro.measure(
          "hunspell_" + name,
          () -> {
            long tokens = 0;
            for (String text : docs) tokens += SweepMicro.consume(a.tokenStream("body", text));
            return tokens;
          });
      a.close();
    }
  }
}
