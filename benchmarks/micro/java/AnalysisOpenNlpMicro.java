import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;

/**
 * M12 T12.5's pairs (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_analysis_opennlp.rs}): analysis-opennlp chains, built by their
 * factories over the models {@code fixtures/src/GenAnalysisOpenNLP.java} trains, over that
 * fixture's text, one document per line. Units are tokens; output is {@code
 * name\tns_per_token\ttokens}.
 */
public class AnalysisOpenNlpMicro {
  static final Path DIR = Path.of("fixtures/data/analysis_opennlp");

  static Analyzer chain(String[]... filters) throws IOException {
    CustomAnalyzer.Builder b =
        CustomAnalyzer.builder(DIR)
            .withTokenizer("openNlp", "sentenceModel", "en-sent.bin", "tokenizerModel", "en-token.bin");
    for (String[] f : filters) {
      String[] args = new String[f.length - 1];
      System.arraycopy(f, 1, args, 0, args.length);
      b.addTokenFilter(f[0], args);
    }
    return b.build();
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
    List<String> docs = new ArrayList<>();
    for (String line : Files.readAllLines(DIR.resolve("opennlp_text.txt"), StandardCharsets.UTF_8)) {
      if (!line.isEmpty()) docs.add(line);
    }
    String[] pos = {"openNlppos", "posTaggerModel", "en-pos-maxent.bin"};
    run("opennlp_tokenizer", chain(), docs);
    run("opennlp_pos", chain(pos), docs);
    run("opennlp_pos_perceptron",
        chain(new String[] {"openNlppos", "posTaggerModel", "en-pos-perceptron.bin"}), docs);
    run("opennlp_chunker", chain(pos, new String[] {"openNlpChunker", "chunkerModel", "en-chunker.bin"}), docs);
    run("opennlp_lemmatizer", chain(pos, new String[] {"openNlpLemmatizer", "dictionary", "en-lemmas.dict",
        "lemmatizerModel", "en-lemmatizer.bin"}), docs);
  }
}
