import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
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

    // Task #207 (full UAX#29-style tokenizer): bare StandardTokenizer output
    // (no stopwords/lowercasing) over strings exercising combining marks,
    // CJK ideograph segmentation, and Hangul syllable clustering, to
    // differentially confirm this port's unicode-segmentation-backed
    // tokenize() agrees with real StandardTokenizer on these cases.
    try (Analyzer plain = new PlainStandardAnalyzer()) {
      // "e" + combining acute accent (U+0301), decomposed "café".
      analyze(m, "uax29_combining_mark", "café today", plain);
      // Four Han ideographs -- each its own token, unlike Latin clustering.
      analyze(m, "uax29_cjk", "你好世界", plain);
      // Precomposed Hangul syllables (single codepoints already).
      analyze(m, "uax29_hangul_precomposed", "안녕하세요", plain);
      // Conjoining Hangul Jamo (leading + vowel + trailing) forming one
      // syllable block: U+1100 U+1161 U+11A8 = "각".
      analyze(m, "uax29_hangul_jamo", "각", plain);
      // Mixed CJK + Latin in one sentence.
      analyze(m, "uax29_mixed_cjk_latin", "hello 世界 world", plain);
      // Midword punctuation: numeric decimal/comma, acronym periods, and an
      // apostrophe contraction, differentially confirmed against real
      // StandardTokenizer's MidNum/MidNumLet/MidLetter rules rather than
      // only this port's own hardcoded-expectation unit tests.
      analyze(m, "uax29_midword_punct", "3.14 U.S.A. don't 1,000", plain);
    }

    // Task #208 (second analyzer-chain producer): real KeywordAnalyzer --
    // whole-field-value-as-one-token, no tokenization/lowercasing/filtering
    // at all -- over a handful of representative inputs (plain id-like
    // string, mixed-case with punctuation that would otherwise split under
    // StandardAnalyzer, a string with embedded whitespace, non-ASCII text,
    // and the empty-string edge case, which KeywordTokenizer still turns
    // into one empty token, not zero (it unconditionally sets its done
    // flag and returns true on the first call regardless of characters
    // read -- see KeywordTokenizer.incrementToken()).
    try (Analyzer keyword = new KeywordAnalyzer()) {
      analyze(m, "keyword_simple", "ID-12345", keyword);
      analyze(m, "keyword_mixed_case_punct", "Status=ACTIVE!", keyword);
      analyze(m, "keyword_whitespace", "  hello world  ", keyword);
      analyze(m, "keyword_non_ascii", "café naïve", keyword);
      analyze(m, "keyword_empty", "", keyword);
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

  /** Bare StandardTokenizer, no filters at all -- raw tokenizer output. */
  static class PlainStandardAnalyzer extends Analyzer {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new StandardTokenizer();
      return new TokenStreamComponents(source);
    }
  }
}
