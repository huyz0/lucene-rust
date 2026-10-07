import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.ar.ArabicAnalyzer;
import org.apache.lucene.analysis.core.FlattenGraphFilter;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.de.GermanAnalyzer;
import org.apache.lucene.analysis.el.GreekAnalyzer;
import org.apache.lucene.analysis.es.SpanishAnalyzer;
import org.apache.lucene.analysis.fr.FrenchAnalyzer;
import org.apache.lucene.analysis.hi.HindiAnalyzer;
import org.apache.lucene.analysis.pt.PortugueseStemFilter;
import org.apache.lucene.analysis.ru.RussianLightStemFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.synonym.SolrSynonymParser;
import org.apache.lucene.analysis.synonym.SynonymFilter;
import org.apache.lucene.analysis.synonym.SynonymGraphFilter;
import org.apache.lucene.analysis.synonym.SynonymMap;

/**
 * M11 part 3's pair (the Rust twin is {@code benchmarks/rust-runner/src/micro_analysis_lang.rs}):
 * each language analyzer over its own language's lines of {@code fixtures/corpus/analysis-lang.txt}
 * (repeated 200 times as documents; the whole multilingual corpus would mostly measure a stemmer
 * missing on foreign words), and synonym filters over {@code fixtures/corpus/analysis-synonym.txt}
 * repeated 50 times. Units are tokens; output is {@code name\tns_per_token\ttokens}.
 */
public class AnalysisLangMicro {
  static List<String> docs(String file) throws IOException {
    List<String> lines = Files.readAllLines(Path.of("fixtures/corpus", file), StandardCharsets.UTF_8);
    List<String> out = new ArrayList<>();
    for (int i = 0; i < 50; i++) out.addAll(lines);
    return out;
  }

  /** The corpus's lines numbered {@code nums} (1-based), repeated 200 times. */
  static List<String> lines(int... nums) throws IOException {
    List<String> all =
        Files.readAllLines(Path.of("fixtures/corpus/analysis-lang.txt"), StandardCharsets.UTF_8);
    List<String> out = new ArrayList<>();
    for (int i = 0; i < 200; i++) for (int n : nums) out.add(all.get(n - 1));
    return out;
  }

  static Analyzer chain(java.util.function.Supplier<Tokenizer> t, java.util.function.Function<TokenStream, TokenStream> f) {
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer src = t.get();
        return new TokenStreamComponents(src, f.apply(src));
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
    List<String> syn = docs("analysis-synonym.txt");
    SolrSynonymParser p = new SolrSynonymParser(true, true, chain(WhitespaceTokenizer::new, LowerCaseFilter::new));
    p.parse(new StringReader(Files.readString(Path.of("fixtures/corpus/synonyms-solr.txt"), StandardCharsets.UTF_8)));
    SynonymMap map = p.build();
    run("german", new GermanAnalyzer(), lines(1, 2));
    run("french", new FrenchAnalyzer(), lines(3, 4));
    run("spanish", new SpanishAnalyzer(), lines(5));
    run("arabic", new ArabicAnalyzer(), lines(28));
    run("hindi", new HindiAnalyzer(), lines(31));
    run("greek", new GreekAnalyzer(), lines(36));
    run("russian_light", chain(StandardTokenizer::new, t -> new RussianLightStemFilter(new LowerCaseFilter(t))), lines(16));
    run("portuguese_rslp", chain(StandardTokenizer::new, t -> new PortugueseStemFilter(new LowerCaseFilter(t))), lines(7));
    run("synonym_graph", chain(WhitespaceTokenizer::new, t -> new SynonymGraphFilter(new LowerCaseFilter(t), map, true)), syn);
    run("synonym_graph_flatten", chain(WhitespaceTokenizer::new, t -> new FlattenGraphFilter(new SynonymGraphFilter(new LowerCaseFilter(t), map, true))), syn);
    run("synonym_legacy", chain(WhitespaceTokenizer::new, t -> new SynonymFilter(new LowerCaseFilter(t), map, true)), syn);
  }
}
