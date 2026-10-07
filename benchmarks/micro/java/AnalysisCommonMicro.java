import java.io.IOException;
import java.io.Reader;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.regex.Pattern;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.charfilter.HTMLStripCharFilter;
import org.apache.lucene.analysis.cjk.CJKAnalyzer;
import org.apache.lucene.analysis.core.SimpleAnalyzer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.email.UAX29URLEmailAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.en.KStemFilter;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.miscellaneous.WordDelimiterGraphFilter;
import org.apache.lucene.analysis.ngram.NGramTokenizer;
import org.apache.lucene.analysis.pattern.PatternTokenizer;
import org.apache.lucene.analysis.shingle.ShingleFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;

/**
 * M11's analysis-common pair (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_analysis_common.rs}): one analyzer per ported package over
 * {@link SweepMicro}'s documents, each token's term bytes read as {@code IndexingChain} reads
 * them. Units are tokens; output is {@code name\tns_per_token\ttokens}.
 */
public class AnalysisCommonMicro {
  static Analyzer chain(
      Function<Reader, Reader> charFilter,
      Supplier<Tokenizer> tokenizer,
      Function<TokenStream, TokenStream> filters) {
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer t = tokenizer.get();
        return new TokenStreamComponents(t, filters.apply(t));
      }

      @Override
      protected Reader initReader(String fieldName, Reader reader) {
        return charFilter.apply(reader);
      }
    };
  }

  static Analyzer chain(Supplier<Tokenizer> tokenizer, Function<TokenStream, TokenStream> filters) {
    return chain(r -> r, tokenizer, filters);
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
    List<String> docs = SweepMicro.analysisDocs();
    List<String> multi = SweepMicro.multilingualDocs();
    List<String> html = docs.stream().map(d -> "<p>" + d + "</p>").toList();
    int wdgf =
        WordDelimiterGraphFilter.GENERATE_WORD_PARTS
            | WordDelimiterGraphFilter.GENERATE_NUMBER_PARTS
            | WordDelimiterGraphFilter.SPLIT_ON_CASE_CHANGE
            | WordDelimiterGraphFilter.SPLIT_ON_NUMERICS
            | WordDelimiterGraphFilter.STEM_ENGLISH_POSSESSIVE;
    run("whitespace", new WhitespaceAnalyzer(), docs);
    run("simple", new SimpleAnalyzer(), docs);
    run("english", new EnglishAnalyzer(), docs);
    run("wdgf", chain(WhitespaceTokenizer::new, t -> new WordDelimiterGraphFilter(t, wdgf, null)), docs);
    run("ngram_2_3", chain(() -> new NGramTokenizer(2, 3), t -> t), docs);
    run("shingle", chain(StandardTokenizer::new, t -> new ShingleFilter(t, 2, 2)), docs);
    run("kstem", chain(StandardTokenizer::new, t -> new KStemFilter(new LowerCaseFilter(t))), docs);
    run("pattern", chain(() -> new PatternTokenizer(Pattern.compile("[ ,.]+"), -1), t -> t), docs);
    run("html_strip", chain(HTMLStripCharFilter::new, WhitespaceTokenizer::new, t -> t), html);
    run("ascii_folding_multilingual", chain(StandardTokenizer::new, ASCIIFoldingFilter::new), multi);
    run("cjk_multilingual", new CJKAnalyzer(), multi);
    run("uax29_url_email_multilingual", new UAX29URLEmailAnalyzer(), multi);
    run(
        "custom_analyzer",
        CustomAnalyzer.builder()
            .withTokenizer("standard")
            .addTokenFilter("lowercase")
            .addTokenFilter("stop")
            .addTokenFilter("porterStem")
            .build(),
        docs);
  }
}
