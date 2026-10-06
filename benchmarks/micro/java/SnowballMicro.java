import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.snowball.SnowballFilter;

/**
 * M11 T11.3's Snowball pair (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_snowball.rs}): {@code WhitespaceTokenizer} + {@code
 * SnowballFilter} per language over that language's fixture vocabulary ({@code
 * fixtures/data/snowball/<Language>.words}, the words only, 100 to a document). Units are tokens;
 * output is {@code name\tns_per_token\ttokens}.
 */
public class SnowballMicro {
  static final String[] LANGUAGES = {"English", "German", "French", "Russian", "Arabic", "Turkish"};

  static List<String> docs(String lang) throws IOException {
    List<String> words = new ArrayList<>();
    for (String line :
        Files.readAllLines(
            Path.of("fixtures/data/snowball/" + lang + ".words"), StandardCharsets.UTF_8)) {
      String w = line.split("\t", -1)[0];
      if (!w.isEmpty() && w.codePoints().noneMatch(Character::isWhitespace)) words.add(w);
    }
    List<String> docs = new ArrayList<>();
    for (int i = 0; i < words.size(); i += 100) {
      docs.add(String.join(" ", words.subList(i, Math.min(words.size(), i + 100))));
    }
    return docs;
  }

  public static void main(String[] args) throws Exception {
    for (String lang : LANGUAGES) {
      List<String> docs = docs(lang);
      Analyzer a =
          new Analyzer() {
            @Override
            protected TokenStreamComponents createComponents(String field) {
              Tokenizer t = new WhitespaceTokenizer();
              return new TokenStreamComponents(t, new SnowballFilter(t, lang));
            }
          };
      SweepMicro.measure(
          "snowball_" + lang.toLowerCase(java.util.Locale.ROOT),
          () -> {
            long tokens = 0;
            for (String text : docs) tokens += SweepMicro.consume(a.tokenStream("body", text));
            return tokens;
          });
      a.close();
    }
  }
}
