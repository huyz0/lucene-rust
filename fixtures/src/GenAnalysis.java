import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.core.KeywordAnalyzer;
import org.apache.lucene.analysis.core.KeywordTokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.ngram.EdgeNGramTokenFilter;
import org.apache.lucene.analysis.ngram.NGramTokenFilter;
import org.apache.lucene.analysis.synonym.SynonymGraphFilter;
import org.apache.lucene.analysis.synonym.SynonymMap;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.util.CharsRef;
import org.apache.lucene.util.CharsRefBuilder;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.snowball.SnowballFilter;
import org.apache.lucene.analysis.fr.FrenchAnalyzer;
import org.tartarus.snowball.ext.EnglishStemmer;
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

    // Task #209 (Porter2/Snowball English stemmer): real Snowball
    // `EnglishStemmer` (the actual `org.tartarus.snowball.ext.EnglishStemmer`
    // generated from Snowball's english.sbl -- the algorithm real Lucene
    // exposes via SnowballFilter(new EnglishStemmer()), a different filter
    // than EnglishAnalyzer's default PorterStemFilter) over a representative
    // word list: base Porter step 1a/1b/1c vocabulary, the full step 2/3/4
    // suffix families (including cases where Porter2's replacement suffix
    // differs from classic Porter's, e.g. "tional"->"tion" not "-tional"
    // unchanged), R1/R2-region-sensitive words, the a_0 exceptional short
    // prefixes ("gener"/"commun"/"arsen"/... that force R1 to start at
    // position 5 rather than after the first vowel-consonant sequence),
    // step 0 apostrophe handling ("don't", "cats'", "'tis", "o'clock"),
    // Porter2's dedicated y/Y "vowel-or-consonant" bookkeeping (double
    // y-initial-vowel words), doubled-consonant undoubling ("controll"->
    // "control", "roll" unchanged since m(word) is not > 1), and the
    // exception1 short-word table ("skis"->"ski", "skies"->"sky", "sky"
    // unchanged, "idly"->"idl", "gently"->"gentl", "ugly"->"ugli",
    // "early"->"earli", "only"->"onli", "singly"->"singl", plus a few
    // invariant words: "andes", "atlas", "bias", "cosmos", "news", "howe").
    try (Analyzer snow = new SnowballEnglishAnalyzer()) {
      String[] words = {
        "caresses", "ponies", "ties", "caress", "cats",
        "feed", "agreed", "plastered", "bled", "motoring", "sing",
        "conflated", "troubled", "sized", "hopping", "tanned", "falling",
        "hissing", "fizzed", "failing", "filing",
        "happy", "sky",
        "relational", "conditional", "rational", "valenci", "hesitanci",
        "digitizer", "conformabli", "radicalli", "differentli", "vileli",
        "analogousli", "vietnamization", "predication", "operator",
        "feudalism", "decisiveness", "hopefulness", "callousness",
        "formaliti", "sensitiviti", "sensibiliti",
        "triplicate", "formative", "formalize", "electriciti", "electrical",
        "hopeful", "goodness",
        "revival", "allowance", "inference", "airliner", "gyroscopic",
        "adjustable", "defensible", "irritant", "replacement", "adjustment",
        "dependent", "adoption", "homologous", "communism", "activate",
        "angulariti", "effective", "bowdlerize", "probate",
        "rate", "cease", "controll", "roll",
        "generalization", "generalize", "generous",
        "arsenal", "commune", "emergency", "lately", "organization",
        "pastime", "universal",
        "proceed", "exceed", "succeed",
        "skis", "skies", "dying", "lying", "tying",
        "idly", "gently", "ugly", "early", "only", "singly",
        "atlas", "bias", "cosmos", "andes", "news", "howe",
        "don't", "doesn't", "cats'", "o'clock", "'tis",
        "syzygy", "toy", "cry"
      };
      analyze(m, "snowball_english", String.join(" ", words), snow);
    }

    // Task #220 (French default stopword list): StandardTokenizer +
    // LowerCaseFilter + real StopFilter fed FrenchAnalyzer.getDefaultStopSet()
    // directly (NOT the full FrenchAnalyzer -- no elision, no French
    // stemming, matching this port's deliberately narrower scope: just the
    // stopword list, wired through this crate's existing language-agnostic
    // StopFilter). Sentence contains several of the 154 default French
    // stopwords ("le", "et", "la", "sont", "dans") plus non-stopword content
    // words, to exercise removal + position-increment carry-over the same
    // way the English case1-case6 cases do.
    try (Analyzer frenchStop = new FrenchStopOnlyAnalyzer()) {
      analyze(m, "french_stopwords", "Le chat et la souris sont dans la maison", frenchStop);
    }

    // b8 sweep (Porter stemmer): real `PorterStemFilter` -- the filter
    // `EnglishAnalyzer` uses by default, and a *different* algorithm from the
    // Snowball English stemmer above. The word list covers the plural/-ed/-ing
    // vocabulary from Porter's own paper, every suffix family in Java's
    // step3/step4/step5 switches (including the `logi -> log` rule and the
    // `bli -> ble` rule this port had wrong), the words where the first
    // matching suffix fails its `m()` test and Java stops rather than falling
    // through ("argument", "ization"), and the one- and two-character words
    // Java's `k > k0 + 1` guard returns untouched (this port used to stem "s"
    // to the empty string). "Cats"/"café" are there because Java's stemmer has
    // no lowercase-ASCII precondition at all.
    try (Analyzer porter = new PorterAnalyzer()) {
      String[] words = {
        "caresses", "ponies", "ties", "caress", "cats", "Cats", "café",
        "feed", "agreed", "disabled", "matting", "mating", "meeting",
        "milling", "messing", "meetings", "plastered", "bled", "motoring",
        "sing", "conflated", "troubled", "sized", "hopping", "tanned",
        "falling", "hissing", "fizzed", "failing", "filing", "happy", "sky",
        "relational", "conditional", "rational", "valenci", "hesitanci",
        "digitizer", "conformabli", "radicalli", "differentli", "vileli",
        "analogousli", "vietnamization", "predication", "operator",
        "feudalism", "decisiveness", "hopefulness", "callousness",
        "formaliti", "sensitiviti", "sensibiliti", "triplicate", "formative",
        "formalize", "electriciti", "electrical", "hopeful", "goodness",
        "revival", "allowance", "inference", "airliner", "gyroscopic",
        "adjustable", "defensible", "irritant", "replacement", "adjustment",
        "dependent", "adoption", "homologous", "communism", "activate",
        "angulariti", "effective", "bowdlerize", "probate", "rate", "cease",
        "controll", "roll", "element", "possibly", "technology", "biology",
        "apology", "methodology", "argument", "ization", "syzygy", "toy",
        "cry", "running", "flies", "happiness",
        "s", "as", "is", "us", "es", "ay", "a"
      };
      analyze(m, "porter_english", String.join(" ", words), porter);
    }

    // b8 sweep (ASCIIFoldingFilter): one token per Unicode block the real
    // filter covers but this port used to miss entirely -- Latin Extended-A,
    // Latin Extended-B, Latin Extended Additional (precomposed Vietnamese),
    // General Punctuation, superscripts, Enclosed Alphanumerics, Latin
    // Extended-C/D, Alphabetic Presentation Forms (ligatures) and
    // Halfwidth/Fullwidth Forms. KeywordTokenizer, so nothing is dropped as
    // punctuation before the filter sees it.
    try (Analyzer fold = new FoldKeywordAnalyzer()) {
      analyze(m, "fold_latin_ext_a", "\u0100\u0110\u0131\u017F\u0132", fold);
      analyze(m, "fold_latin_ext_b", "\u0180\u01C4\u0222\u024F", fold);
      analyze(m, "fold_vietnamese", "\u1EBF\u1EAB\u1EB7\u1ED9\u1EEF", fold);
      analyze(m, "fold_punctuation", "\u00AB\u00BB\u2018\u2019\u201C\u201D\u2013\u2014\u2026", fold);
      analyze(m, "fold_superscripts", "\u00B2\u00B3\u00B9\u2070\u2081", fold);
      analyze(m, "fold_enclosed_alnum", "\u2460\u24B6\u24D0", fold);
      analyze(m, "fold_ligatures", "\uFB00\uFB01\uFB02\uFB05\uFB06", fold);
      analyze(m, "fold_fullwidth", "\uFF21\uFF41\uFF10\uFF5B", fold);
    }

    // b8 sweep (LowerCaseFilter): Java lowercases with
    // `Character.toLowerCase(codePoint)`, the *simple* 1:1 mapping, written
    // back in place. "\u0130" folds to a single 'i' (not "i" + combining dot
    // above), and a word-final Greek sigma stays a plain sigma (no
    // final-sigma rule), unlike full Unicode lowercasing.
    try (Analyzer lower = new LowerKeywordAnalyzer()) {
      analyze(m, "lowercase_dotted_capital_i", "\u0130", lower);
      analyze(m, "lowercase_greek_final_sigma", "\u039F\u0394\u039F\u03A3", lower);
      analyze(m, "lowercase_german_sharp_s", "\u1E9E", lower);
    }

    // b8 sweep (n-gram filters): the position-increment carry across a token
    // too short to produce any gram, and the fact that every gram keeps the
    // *input token's* offsets (Java `restoreState`s and never calls
    // setOffset).
    try (Analyzer ng = new NGramAnalyzer(3, 3, false)) {
      analyze(m, "ngram_skipped_short_token", "big a cat", ng);
    }
    try (Analyzer ng = new NGramAnalyzer(2, 3, false)) {
      analyze(m, "ngram_offsets_are_the_input_tokens", "abcde", ng);
    }
    try (Analyzer ng = new NGramAnalyzer(2, 3, true)) {
      analyze(m, "ngram_preserve_original", "a abcd", ng);
    }
    try (Analyzer ng = new EdgeNGramAnalyzer(2, 3, false)) {
      analyze(m, "edge_ngram_basic", "abcde", ng);
    }
    try (Analyzer ng = new EdgeNGramAnalyzer(2, 3, true)) {
      analyze(m, "edge_ngram_preserve_original", "a abcd", ng);
    }

    // b8 sweep (SynonymGraphFilter): emission order, position increments and
    // position lengths come out of the synonym graph's nodes, and the
    // original tokens keep their own offsets. Recorded with
    // positionLength, which the other cases here do not need.
    analyzeGraph(m, "syn_multiword_to_single", "wi fi", new String[][] {{"wi fi", "wifi"}});
    analyzeGraph(m, "syn_two_alternatives", "wi fi", new String[][] {{"wi fi", "wifi", "wireless"}});
    analyzeGraph(m, "syn_multiword_to_multiword", "new york", new String[][] {{"new york", "big apple"}});
    analyzeGraph(m, "syn_single_to_multiword", "usa", new String[][] {{"usa", "united states of america"}});
    analyzeGraph(
        m, "syn_in_context", "the wi fi router", new String[][] {{"wi fi", "wifi"}});
    analyzeGraph(m, "syn_single_to_single", "the cat sat", new String[][] {{"cat", "feline"}});

    // c33 sweep (the offset *unit*): OffsetAttribute reports **UTF-16 code
    // unit** (Java `char`) indices into the original String -- not UTF-8
    // bytes and not Unicode scalars. Every text below separates all three,
    // so a producer using the wrong unit cannot pass:
    //
    //   character            scalars  UTF-16  UTF-8
    //   ASCII 'a'                  1       1      1
    //   'e\u0301' / '\u00E9'        1/2*     1/2*    2/3*
    //   CJK '\u4E16'                1       1      3
    //   astral '\uD83D\uDE00'        1       2      4
    //
    // (*combining sequences are two scalars/two code units/three bytes.)
    try (Analyzer plain = new PlainStandardAnalyzer()) {
      // Precomposed Latin-1: 1 char, 2 UTF-8 bytes.
      analyze(m, "utf16_latin1", "caf\u00E9 dog", plain);
      // CJK: 1 char, 3 UTF-8 bytes, and each ideograph its own token.
      analyze(m, "utf16_cjk_offsets", "abc \u4E16\u754C def", plain);
      // Emoji: 2 chars, 1 scalar, 4 UTF-8 bytes. Real StandardTokenizer
      // emits it as a token of its own (b8's F40: this port's
      // `unicode_word_indices` does not), so what this case pins here is
      // that the emoji shifts every later token by exactly 2 chars.
      analyze(m, "utf16_emoji", "alpha \uD83D\uDE00 beta", plain);
      // TETRAGRAM FOR CENTRE: supplementary-plane and *not* pictographic, so
      // neither engine emits a token for it -- an astral shift with no F40
      // interaction, compared verbatim.
      analyze(m, "utf16_astral_symbol", "alpha \uD834\uDF06 beta", plain);
      // A supplementary-plane *letter* (MATHEMATICAL BOLD CAPITAL A/B), so
      // the astral run is a token whose own span is 2 chars per scalar.
      analyze(m, "utf16_astral_letter", "x \uD835\uDC00\uD835\uDC01 y", plain);
      // Decomposed combining mark: the token is 5 chars / 6 bytes.
      analyze(m, "utf16_combining_mark_offsets", "cafe\u0301 dog", plain);
      // All of them in one text, in one pass.
      analyze(
          m,
          "utf16_all_units",
          "alpha caf\u00E9 \u4E16 \uD834\uDF06 e\u0301x \uD835\uDC00 omega",
          plain);
    }

    // c33: KeywordTokenizer's single token ends at `correctOffset(charCount)`
    // -- a Java char count, so an astral input's end offset exceeds its
    // scalar count and falls short of its byte count.
    try (Analyzer keyword = new KeywordAnalyzer()) {
      analyze(m, "utf16_keyword_astral", "id-\uD83D\uDE00-\u00E9", keyword);
    }

    // c33: a filter that *changes the term's length* must still report the
    // source text's span. ASCIIFoldingFilter grows "stra\u00DFe" (6 chars) to
    // "strasse" (7), after an emoji that has already shifted the offsets.
    try (Analyzer foldOnly = new FoldOnlyAnalyzer()) {
      analyze(m, "utf16_fold_after_astral", "\uD835\uDC00 stra\u00DFe", foldOnly);
    }

    // c33: same for PorterStemFilter ("running" -> "run"), and the astral
    // token itself passes through the stemmer untouched.
    try (Analyzer porter = new PorterAnalyzer()) {
      analyze(m, "utf16_porter_after_astral", "\uD835\uDC00 running fishes", porter);
    }

    // c33: the n-gram filters restoreState() the input token's offsets, so
    // every gram of a non-ASCII token reports that token's char span.
    try (Analyzer ng = new NGramAnalyzer(2, 3, false)) {
      analyze(m, "utf16_ngram_offsets", "caf\u00E9 \uD835\uDC00\uD835\uDC01cd", ng);
    }
    try (Analyzer ng = new EdgeNGramAnalyzer(2, 3, false)) {
      analyze(m, "utf16_edge_ngram_offsets", "\uD835\uDC00bc d\u00E9f", ng);
    }

    // c33: SynonymGraphFilter's collapsed match spans from the first matched
    // token's startOffset to the last one's endOffset, in char units, and the
    // originals keep their own -- the non-decreasing-startOffset rule b8 fixed
    // has to keep holding once the unit changes.
    analyzeGraph(
        m,
        "utf16_syn_multiword",
        "\uD835\uDC00 wi fi \u4E16",
        new String[][] {{"wi fi", "wifi"}});

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

  /**
   * StandardTokenizer + LowerCaseFilter + StopFilter(FrenchAnalyzer's default
   * stop set) -- task #220's French default stopword list, deliberately
   * *not* the full FrenchAnalyzer (no ElisionFilter, no French stemming).
   */
  static class FrenchStopOnlyAnalyzer extends Analyzer {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new StandardTokenizer();
      TokenStream filter = new LowerCaseFilter(source);
      filter = new StopFilter(filter, FrenchAnalyzer.getDefaultStopSet());
      return new TokenStreamComponents(source, filter);
    }
  }

  /**
   * StandardTokenizer + LowerCaseFilter + SnowballFilter(EnglishStemmer) --
   * task #209's Porter2/Snowball English stemmer, a different filter than
   * EnglishAnalyzer's default (classic Porter) PorterStemFilter.
   */
  static class SnowballEnglishAnalyzer extends Analyzer {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new StandardTokenizer();
      TokenStream filter = new LowerCaseFilter(source);
      filter = new SnowballFilter(filter, new EnglishStemmer());
      return new TokenStreamComponents(source, filter);
    }
  }

  /** KeywordTokenizer + PorterStemFilter -- one token per input word. */
  static class PorterAnalyzer extends Analyzer {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new WhitespaceTokenizer();
      return new TokenStreamComponents(source, new PorterStemFilter(source));
    }
  }

  /** KeywordTokenizer + ASCIIFoldingFilter -- nothing dropped as punctuation. */
  static class FoldKeywordAnalyzer extends Analyzer {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new KeywordTokenizer();
      return new TokenStreamComponents(source, new ASCIIFoldingFilter(source));
    }
  }

  /** KeywordTokenizer + LowerCaseFilter. */
  static class LowerKeywordAnalyzer extends Analyzer {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new KeywordTokenizer();
      return new TokenStreamComponents(source, new LowerCaseFilter(source));
    }
  }

  /** WhitespaceTokenizer + NGramTokenFilter. */
  static class NGramAnalyzer extends Analyzer {
    final int min, max;
    final boolean preserve;

    NGramAnalyzer(int min, int max, boolean preserve) {
      this.min = min;
      this.max = max;
      this.preserve = preserve;
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new WhitespaceTokenizer();
      return new TokenStreamComponents(
          source, new NGramTokenFilter(source, min, max, preserve));
    }
  }

  /** WhitespaceTokenizer + EdgeNGramTokenFilter. */
  static class EdgeNGramAnalyzer extends Analyzer {
    final int min, max;
    final boolean preserve;

    EdgeNGramAnalyzer(int min, int max, boolean preserve) {
      this.min = min;
      this.max = max;
      this.preserve = preserve;
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new WhitespaceTokenizer();
      return new TokenStreamComponents(
          source, new EdgeNGramTokenFilter(source, min, max, preserve));
    }
  }

  /**
   * Records a {@link SynonymGraphFilter} run, including {@code
   * PositionLengthAttribute} (which the other cases do not need). {@code rules}
   * is one row per rule: element 0 is the space-separated input phrase, the
   * rest are the space-separated output phrases.
   */
  static void analyzeGraph(StringBuilder m, String caseName, String text, String[][] rules)
      throws IOException {
    SynonymMap.Builder b = new SynonymMap.Builder(true);
    for (String[] rule : rules) {
      for (int i = 1; i < rule.length; i++) {
        b.add(joinPhrase(rule[0]), joinPhrase(rule[i]), true);
      }
    }
    SynonymMap map;
    try {
      map = b.build();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    StringBuilder tokensOut = new StringBuilder();
    int count = 0;
    try (Analyzer analyzer = new SynonymGraphAnalyzer(map);
        TokenStream ts = analyzer.tokenStream("field", text)) {
      CharTermAttribute termAtt = ts.addAttribute(CharTermAttribute.class);
      OffsetAttribute offsetAtt = ts.addAttribute(OffsetAttribute.class);
      PositionIncrementAttribute posIncAtt = ts.addAttribute(PositionIncrementAttribute.class);
      PositionLengthAttribute posLenAtt = ts.addAttribute(PositionLengthAttribute.class);
      ts.reset();
      while (ts.incrementToken()) {
        if (tokensOut.length() > 0) tokensOut.append(';');
        tokensOut
            .append(termAtt.toString())
            .append(':')
            .append(posIncAtt.getPositionIncrement())
            .append(':')
            .append(posLenAtt.getPositionLength())
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

  static CharsRef joinPhrase(String phrase) {
    return SynonymMap.Builder.join(phrase.split(" "), new CharsRefBuilder());
  }

  /** WhitespaceTokenizer + SynonymGraphFilter. */
  static class SynonymGraphAnalyzer extends Analyzer {
    final SynonymMap map;

    SynonymGraphAnalyzer(SynonymMap map) {
      this.map = map;
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new WhitespaceTokenizer();
      return new TokenStreamComponents(source, new SynonymGraphFilter(source, map, true));
    }
  }
}
