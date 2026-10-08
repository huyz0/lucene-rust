import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import com.ibm.icu.text.Collator;
import com.ibm.icu.text.RuleBasedCollator;
import com.ibm.icu.util.ULocale;
import org.apache.lucene.analysis.icu.ICUCollationKeyAnalyzer;
import org.apache.lucene.analysis.icu.ICUFoldingFilter;
import org.apache.lucene.analysis.icu.ICUNormalizer2CharFilter;
import org.apache.lucene.analysis.icu.ICUNormalizer2Filter;
import org.apache.lucene.analysis.icu.segmentation.ICUTokenizer;

/**
 * M12 T12.4's pairs (the Rust twin is {@code benchmarks/rust-runner/src/micro_analysis_icu.rs}):
 * analysis-icu over {@code fixtures/corpus/analysis-icu.txt} and the normalization fixture's
 * stress strings, one document per line. Units are tokens; output is {@code
 * name\tns_per_token\ttokens}.
 */
public class AnalysisIcuMicro {
  /** Every non-empty line of the corpus and every unescaped stress string. */
  static List<String> docs() throws IOException {
    List<String> docs = new ArrayList<>();
    for (String line : Files.readAllLines(Path.of("fixtures/corpus/analysis-icu.txt"), StandardCharsets.UTF_8)) {
      if (!line.isEmpty()) docs.add(line);
    }
    for (String line : Files.readAllLines(Path.of("fixtures/data/analysis_icu/norm_strings.txt"), StandardCharsets.UTF_8)) {
      if (!line.isEmpty() && line.indexOf('\\') < 0) docs.add(line);
    }
    return docs;
  }

  static Analyzer chain(Supplier<Tokenizer> tok, Function<TokenStream, TokenStream> filters) {
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer t = tok.get();
        return new TokenStreamComponents(t, filters.apply(t));
      }
    };
  }

  static void run(String name, Analyzer a, List<String> docs) throws IOException {
    SweepMicro.measure(
        name,
        () -> {
          long tokens = 0;
          for (String text : docs) tokens += SweepMicro.consume(a.tokenStream("body", text));
          return tokens;
        });
    a.close();
  }

  public static void main(String[] args) throws Exception {
    List<String> docs = docs();
    run("icu_normalizer_nfkc_cf", chain(WhitespaceTokenizer::new, ICUNormalizer2Filter::new), docs);
    run("icu_folding", chain(WhitespaceTokenizer::new, ICUFoldingFilter::new), docs);
    run("icu_normalizer_charfilter", new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        return new TokenStreamComponents(new WhitespaceTokenizer());
      }

      @Override
      protected java.io.Reader initReader(String fieldName, java.io.Reader reader) {
        return new ICUNormalizer2CharFilter(reader);
      }
    }, docs);
    run("icu_tokenizer", chain(ICUTokenizer::new, t -> t), docs);
    run("icu_collation_key", new ICUCollationKeyAnalyzer(Collator.getInstance(ULocale.ROOT)), docs);
    RuleBasedCollator identical = (RuleBasedCollator) Collator.getInstance(new ULocale("de@collation=phonebook"));
    identical.setStrength(Collator.IDENTICAL);
    identical.setAlternateHandlingShifted(true);
    run("icu_collation_key_phonebook_identical", new ICUCollationKeyAnalyzer(identical), docs);
  }
}
