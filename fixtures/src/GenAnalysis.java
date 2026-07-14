import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.CharArraySet;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Generates a differential-testing fixture for {@code lucene-analysis}: runs
 * real Lucene's {@link StandardAnalyzer} (StandardTokenizer + LowerCaseFilter
 * + StopFilter) over a handful of strings with a real stopword set and
 * records the resulting (term, position, offset) triples. This is the
 * single most valuable check for task #61's position-increment-preservation
 * rule in StopFilter: a removed stopword's own increment must be carried
 * onto the next surviving token, not dropped.
 */
public class GenAnalysis {

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("analysis");
    Files.createDirectories(out);

    CharArraySet stopwords = new CharArraySet(Arrays.asList("the", "a", "of"), false);

    StringBuilder m = new StringBuilder();

    // Case 1: matches task's spec example exactly -- "the quick fox" with
    // "the" a stopword.
    analyze(m, "case1", "the quick fox", stopwords);

    // Case 2: stopword at the very start.
    analyze(m, "case2", "the fox", stopwords);

    // Case 3: stopword at the very end.
    analyze(m, "case3", "fox the", stopwords);

    // Case 4: consecutive stopwords in a row.
    analyze(m, "case4", "a the of fox", stopwords);

    // Case 5: text with only stopwords -- empty output.
    analyze(m, "case5", "the a of", stopwords);

    // Case 6: punctuation + mixed case, multi-word sentence -- "The" is
    // itself a stopword (lowercased before the stopword check), so this
    // exercises tokenizer + lowercasing + stopword removal all together.
    analyze(m, "case6", "The Quick, Brown FOX!", stopwords);

    // Task #64 (ASCIIFoldingFilter): a real ASCIIFoldingFilter run, fold-only
    // (no lowercasing), over a string with several diacritics and a
    // ligature -- this checks this port's AsciiFoldingFilter::apply in
    // isolation (case preserved, offsets untouched despite the ligature
    // growing the term's character length).
    try (Analyzer foldOnly = new FoldOnlyAnalyzer()) {
      analyze(m, "fold_only", "café naïve Müller cœur straße", foldOnly);
    }

    // Task #64: the composed Analyzer chain this port wires up via
    // Analyzer::with_ascii_folding -- fold, then lowercase, then (no
    // stopwords here) -- over the same text.
    try (Analyzer foldLower = new FoldThenLowerAnalyzer()) {
      analyze(m, "fold_then_lower", "Café Naïve ÉCOLE", foldLower);
    }

    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote analysis/ fixture directory");
  }

  static void analyze(StringBuilder m, String caseName, String text, CharArraySet stopwords)
      throws IOException {
    try (Analyzer analyzer = new StandardAnalyzer(stopwords)) {
      analyze(m, caseName, text, analyzer);
    }
  }

  static void analyze(StringBuilder m, String caseName, String text, Analyzer analyzer)
      throws IOException {
    StringBuilder tokensOut = new StringBuilder();
    int count = 0;
    try (TokenStream ts = analyzer.tokenStream("field", text)) {
      CharTermAttribute termAtt = ts.addAttribute(CharTermAttribute.class);
      OffsetAttribute offsetAtt = ts.addAttribute(OffsetAttribute.class);
      PositionIncrementAttribute posIncAtt = ts.addAttribute(PositionIncrementAttribute.class);
      ts.reset();
      while (ts.incrementToken()) {
        if (tokensOut.length() > 0) tokensOut.append(';');
        tokensOut
            .append(termAtt.toString())
            .append(':')
            .append(posIncAtt.getPositionIncrement())
            .append(':')
            .append(offsetAtt.startOffset())
            .append(',')
            .append(offsetAtt.endOffset());
        count++;
      }
      ts.end();
    }
    m.append(caseName).append(".text=").append(text).append('\n');
    m.append(caseName).append(".count=").append(count).append('\n');
    m.append(caseName).append(".tokens=").append(tokensOut).append('\n');
  }

  /** StandardTokenizer + ASCIIFoldingFilter only, no lowercasing. */
  static class FoldOnlyAnalyzer extends Analyzer {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new StandardTokenizer();
      TokenStream filter = new ASCIIFoldingFilter(source);
      return new TokenStreamComponents(source, filter);
    }
  }

  /** StandardTokenizer + ASCIIFoldingFilter + LowerCaseFilter, in that order. */
  static class FoldThenLowerAnalyzer extends Analyzer {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new StandardTokenizer();
      TokenStream filter = new ASCIIFoldingFilter(source);
      filter = new LowerCaseFilter(filter);
      return new TokenStreamComponents(source, filter);
    }
  }
}
